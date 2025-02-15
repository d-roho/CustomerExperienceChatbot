import re
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from anthropic import Anthropic

class VectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone vector store."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1536  # Claude embeddings dimension

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
        """Upload text chunks to Pinecone."""
        try:
            print(f"Getting embeddings for {len(texts)} texts")
            embeddings = client.get_embeddings(texts)

            # Prepare vectors for upload
            vectors = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                vectors.append((
                    str(i),
                    embedding,
                    {"text": text}
                ))

            # Upsert to Pinecone
            print(f"Upserting {len(vectors)} vectors to Pinecone")
            self.index.upsert(vectors=vectors)
            print("Successfully upserted vectors")
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

            return [
                {
                    "text": match.metadata["text"],
                    "score": match.score
                }
                for match in results.matches
            ]
        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def rerank_results(self, query: str, results: List[Dict[str, Any]], 
                      client: Anthropic) -> List[Dict[str, Any]]:
        """Rerank results using semantic similarity."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates the relevance of text passages to a query. Rate each passage's relevance from 0 to 1."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRate the relevance of each passage:\n" + 
                          "\n".join([f"Passage {i+1}: {r['text']}" for i, r in enumerate(results)])
            }
        ]

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=messages
        )

        # Extract scores from response
        try:
            scores = [float(score) for score in re.findall(r"(\d*\.?\d+)", response.content)]

            # Combine original and semantic scores
            for i, result in enumerate(results):
                result["combined_score"] = (result["score"] + scores[i]) / 2

            # Sort by combined score
            results.sort(key=lambda x: x["combined_score"], reverse=True)

        except Exception:
            # Fall back to original scoring if parsing fails
            pass

        return results