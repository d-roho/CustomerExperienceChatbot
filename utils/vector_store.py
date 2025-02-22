import re
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from anthropic import Anthropic
# from sentence_transformers import SentenceTransformer
# import torch
from utils.db import MotherDuckStore  # New import
import datetime
import asyncio
import itertools
import utils.vector_funcs


class VectorStore:

    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone vector store and MotherDuck store."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.db = MotherDuckStore()  # Initialize MotherDuck store

        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=api_key)
            print(
                f"Pinecone initialized. Available indexes: {self.pc.list_indexes().names()}"
            )

            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating new index: {self.index_name}")
                try:
                    self.pc.create_index(name=self.index_name,
                                         dimension=self.dimension,
                                         metric='cosine',
                                         spec=ServerlessSpec(
                                             cloud='aws',
                                             region=self.environment))
                    print(f"Successfully created index: {self.index_name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to create Pinecone index: {str(e)}")

            self.index = self.pc.Index(self.index_name)
            print(f"Successfully connected to index: {self.index_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def upsert_texts(self, chunks: List[Dict[str, Any]], client: Anthropic,
                     index_name, df) -> None:
        """Upload text chunks with metadata to Pinecone and MotherDuck."""
        try:
            print(f"\nStarting embedding process for {len(chunks)} chunks")

            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            print("Average chunk size:",
                  sum(len(t) for t in texts) / len(texts), "characters")
            print("Largest chunk size:", max(len(t) for t in texts),
                  "characters")

            print("\nGenerating embeddings...")
            embeddings = client.get_embeddings(texts)
            print(f"Successfully generated {len(embeddings)} embeddings")

            # Store texts in MotherDuck DB
            print("\nStoring texts in MotherDuck DB...")
            self.db.create_table(index_name, df)
            print("Successfully stored all chunks in MotherDuck")

            # Prepare vectors with metadata
            print("\nPreparing vectors for Pinecone upload...")
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append(
                    (chunk['metadata']['id'], embedding, chunk['metadata']))

            # Batch upsert to Pinecone
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            print(
                f"\nStarting batch upload process ({total_batches} batches)..."
            )

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                current_batch = i // batch_size + 1
                print(
                    f"\nUpserting batch {current_batch}/{total_batches} ({len(batch)} vectors)"
                )
                self.index.upsert(vectors=batch)
                print(
                    f"Batch {current_batch} complete - {i + len(batch)}/{len(vectors)} vectors processed"
                )

            print("\nVector upload complete!")
            print(
                f"Successfully processed {len(vectors)} vectors across {total_batches} batches"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to upsert texts: {str(e)}")

    def search(self,
               query: str,
               client: Anthropic,
               top_k: int = 5,
               index_name: str = 'reviews-csv-main') -> List[Dict[str, Any]]:
        """Search for similar texts using Pinecone and retrieve full texts from MotherDuck."""

        try:
            print(f"Getting embedding for query: {query[:50]}...")
            query_embedding = client.get_embeddings([query])[0]
            print(f"Searching Pinecone with top_k={top_k}")
            results = self.index.query(vector=query_embedding,
                                       top_k=top_k,
                                       include_metadata=True)

            print(f"Successfully retrieved {len(results['matches'])} results")
            # Retrieve full texts and metadata from MotherDuck
            processed_results = []
            for match in results.matches:
                stored_data = self.db.get_chunk(match.id, index_name)
                if stored_data:
                    processed_results.append({
                        'text':
                        stored_data['text'],
                        'md_metadata':
                        stored_data['metadata'],
                        'pc_metadata':
                        match['metadata'],
                        'header':
                        match.metadata['header'],
                        'score':
                        match.score
                    })

            return processed_results

        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    async def filter_search(
        self,
        filters: Dict[str, Any],
        query: str,
        client: Anthropic,
        top_k: int = 5, subdivide_k: bool = False,
        index_name: str = 'reviews-csv-main',
    ) -> List[Dict[str, Any]]:
        """Search for similar texts using METADATA FILTERS."""

        try:
            filters, message = utils.vector_funcs.hierarchy_upholder(filters)

            print(f"Filter: {filters} \n {message}")
            reviews_dict = {}
            print(f"Getting embedding for query: {query[:50]}...")
            query_embedding = client.get_embeddings([query])[0]
            print(f"Searching Pinecone with top_k={top_k}")

            subset_combinations, has_date = utils.vector_funcs.subset_generator(filters)
            if subdivide_k: # making Top K proportionate to no. of subsets
                top_k= top_k//len(subset_combinations)
                
            # Create tasks for parallel processing
            tasks = [
                utils.vector_funcs.process_combination(self, query_embedding, top_k, index_name, filters, combo_idx, combo, has_date)
                for combo_idx, combo in enumerate(subset_combinations)
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # Combine results into reviews_dict
            reviews_dict = {
                result['combo_idx']: {
                    'subset_info': result['subset_info'],
                    'processed_results': result['processed_results']
                }
                for result in results
            }

            return reviews_dict

        except Exception as e:
            raise RuntimeError(f"Failed to filter search: {str(e)}")

    async def rerank_results(
            self, query: str,
            results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using semantic similarity with async batching."""
        try:
            from sentence_transformers import CrossEncoder
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Warning: Reranking model not available: {str(e)}")
            return results

        try:
            # Prepare pairs for reranking
            pairs = []
            for result in results:
                text = f"{result['header']}\n{result['text']}"
                pairs.append([query, text])

            # Define batch processing with concurrency limit
            BATCH_SIZE = 20
            MAX_CONCURRENT = 5
            semaphore = asyncio.Semaphore(MAX_CONCURRENT)

            async def process_batch(batch_pairs):
                async with semaphore:
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as pool:
                        return await loop.run_in_executor(
                            pool, model.predict, batch_pairs)

            # Process batches concurrently
            tasks = []
            for i in range(0, len(pairs), BATCH_SIZE):
                batch = pairs[i:i + BATCH_SIZE]
                tasks.append(process_batch(batch))

            batch_scores = await asyncio.gather(*tasks)
            scores = [score for batch in batch_scores for score in batch]

            # Update scores
            for i, result in enumerate(results):
                if i < len(scores):
                    result['score'] = float(scores[i])

            # Sort by new scores
            results.sort(key=lambda x: x.get('score', 0), reverse=True)
            return results
        except Exception as e:
            print(f"Warning: Reranking failed: {str(e)}")
            return results

    def fetch_all_reviews(self, index: str):
        """Fetch all reviews from a MD table ."""
        try:
            reviews = self.db.fetch_all(self.index_name)
        except Exception as e:
            print(f"Warning: Table not retrieved: {str(e)}")

        return reviews
