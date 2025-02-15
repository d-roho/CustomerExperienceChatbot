import re
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
import torch

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

        # Initialize Pinecone with new API
        try:
            self.pc = Pinecone(api_key=api_key)
            print(f"Pinecone initialized. Available indexes: {self.pc.list_indexes().names()}")

            # Create index if it doesn't exist
            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating new index: {self.index_name}")
                try:
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',  # Required for serverless
                            region=self.environment
                        )
                    )
                    print(f"Successfully created index: {self.index_name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to create Pinecone index: {str(e)}")

            # Wait briefly for index to be ready
            import time
            time.sleep(5)

            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            print(f"Successfully connected to index: {self.index_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def upsert_texts(self, texts: List[str], client: Anthropic) -> None:
        """Upload text chunks to Pinecone and Replit DB."""
        try:
            from replit import db

            print(f"\nStarting embedding process for {len(texts)} texts")
            print("Average chunk size:", sum(len(t) for t in texts) / len(texts), "characters")
            print("Largest chunk size:", max(len(t) for t in texts), "characters")

            print("\nGenerating embeddings...")
            embeddings = client.get_embeddings(texts)
            print(f"Successfully generated {len(embeddings)} embeddings")

            print("\nStoring texts in Replit DB...")
            for i, text in enumerate(texts):
                db[f"text_{i}"] = text
            print("Successfully stored texts in DB")

            print("\nPreparing vectors for Pinecone upload...")
            vectors = []
            for i, embedding in enumerate(embeddings):
                vectors.append((
                    str(i),
                    embedding,
                    {}
                ))
            print(f"Prepared {len(vectors)} vectors")

            # Batch upsert to Pinecone
            batch_size = 100  # Smaller batch size to stay under 4MB limit
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            print(f"\nStarting batch upload process ({total_batches} batches)...")

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                current_batch = i//batch_size + 1
                print(f"\nUpserting batch {current_batch}/{total_batches} ({len(batch)} vectors)")

                # Print random chunk from this batch
                from random import choice
                random_id = int(choice(batch)[0])  # Get ID from random vector in batch
                from replit import db
                print(f"Random chunk from batch {current_batch}:")
                print("-" * 50)
                print(db.get(f"text_{random_id}"))
                print("-" * 50)

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

            from replit import db
            return [
                {
                    "text": db.get(f"text_{match.id}"),
                    "score": match.score
                }
                for match in results.matches
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      client: Anthropic) -> List[Dict[str, Any]]:
        """Rerank results using Jina ColBERT v2."""
        try:
            if not self.rerank_model:
                print("Warning: Reranking model not available, falling back to original scores")
                return results

            # Extract texts for reranking
            texts = [result["text"] for result in results]

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
                result["combined_score"] = (result["score"] + similarity) / 2

            # Sort by combined score
            results.sort(key=lambda x: x["combined_score"], reverse=True)

            return results

        except Exception as e:
            print(f"Warning: Reranking failed: {str(e)}")
            return results  # Fall back to original ranking