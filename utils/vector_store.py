import re
from typing import List, Dict, Any
import pinecone
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import torch
from replit import db

class VectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone vector store."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1536  # OpenAI ada-002 embedding dimension

        # Initialize reranking model
        try:
            self.rerank_model = SentenceTransformer('jinaai/jina-colbert-v2')
            self.rerank_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            print("Initialized Jina ColBERT v2 model for reranking")
        except Exception as e:
            print(f"Warning: Could not initialize reranking model: {str(e)}")
            self.rerank_model = None

        # Initialize Pinecone
        try:
            pinecone.init(api_key=api_key, environment=environment)
            print(f"Pinecone initialized. Available indexes: {pinecone.list_indexes()}")

            if self.index_name not in pinecone.list_indexes():
                print(f"Creating new index: {self.index_name}")
                try:
                    pinecone.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric='cosine'
                    )
                    print(f"Successfully created index: {self.index_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to create Pinecone index: {str(e)}")

            self.index = pinecone.Index(self.index_name)
            print(f"Successfully connected to index: {self.index_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def upsert_texts(self, chunks: List[Dict[str, Any]], client: Anthropic) -> None:
        """Upload text chunks with metadata to Pinecone and Replit DB."""
        try:
            print(f"\nStarting embedding process for {len(chunks)} chunks")

            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            print("Average chunk size:", sum(len(t) for t in texts) / len(texts), "characters")
            print("Largest chunk size:", max(len(t) for t in texts), "characters")

            print("\nGenerating embeddings...")
            embeddings = client.get_embeddings(texts)
            print(f"Successfully generated {len(embeddings)} embeddings")

            # Store texts in Replit DB
            print("\nStoring texts in Replit DB...")
            for chunk in chunks:
                chunk_id = chunk['metadata']['id']
                db[f"text_{chunk_id}"] = {
                    'text': chunk['text'],
                    'metadata': chunk['metadata']
                }

            # Prepare vectors with metadata
            print("\nPreparing vectors for Pinecone upload...")
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append((
                    chunk['metadata']['id'],
                    embedding,
                    chunk['metadata']
                ))

            # Batch upsert to Pinecone
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            print(f"\nStarting batch upload process ({total_batches} batches)...")

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                current_batch = i//batch_size + 1
                print(f"\nUpserting batch {current_batch}/{total_batches} ({len(batch)} vectors)")
                self.index.upsert(vectors=batch)
                print(f"Batch {current_batch} complete - {i + len(batch)}/{len(vectors)} vectors processed")

            print("\nVector upload complete!")
            print(f"Successfully processed {len(vectors)} vectors across {total_batches} batches")

        except Exception as e:
            raise RuntimeError(f"Failed to upsert texts: {str(e)}")

    def search(self, query: str, client: Anthropic, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts in Pinecone."""
        try:
            print(f"Getting embedding for query: {query[:50]}...")
            query_embedding = client.get_embeddings([query])[0]

            print(f"Searching Pinecone with top_k={top_k}")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Retrieve full texts and metadata from Replit DB
            processed_results = []
            for match in results.matches:
                stored_data = db.get(f"text_{match.id}")
                if stored_data:
                    processed_results.append({
                        'text': stored_data['text'],
                        'metadata': stored_data['metadata'],
                        'score': match.score
                    })

            return processed_results

        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using Jina ColBERT v2."""
        try:
            if not self.rerank_model:
                print("Warning: Reranking model not available, falling back to original scores")
                return results

            # Extract texts for reranking
            texts = [result['text'] for result in results]

            # Get embeddings for query and texts
            query_embedding = self.rerank_model.encode(query, convert_to_tensor=True)
            text_embeddings = self.rerank_model.encode(texts, convert_to_tensor=True)

            # Calculate cosine similarities
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), 
                text_embeddings
            ).tolist()

            # Update scores and sort
            for result, similarity in zip(results, similarities):
                # Combine original score with reranking score
                result['score'] = (result['score'] + similarity) / 2

            # Sort by combined score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results

        except Exception as e:
            print(f"Warning: Reranking failed: {str(e)}")
            return results  # Fall back to original ranking