import pinecone
from typing import List, Dict, Any
import numpy as np
from anthropic import Anthropic

class VectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone vector store."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1536  # Claude embeddings dimension
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=self.dimension,
                metric='cosine'
            )
        
        self.index = pinecone.Index(index_name)

    def get_embeddings(self, texts: List[str], client: Anthropic) -> List[List[float]]:
        """Get embeddings for texts using Claude."""
        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                model="claude-3-5-sonnet-20241022",
                input=text
            )
            embeddings.append(response.embeddings[0])
        return embeddings

    def upsert_texts(self, texts: List[str], client: Anthropic) -> None:
        """Upload text chunks to Pinecone."""
        embeddings = self.get_embeddings(texts, client)
        
        # Prepare vectors for upload
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vectors.append((
                str(i),
                embedding,
                {"text": text}
            ))
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)

    def search(self, query: str, client: Anthropic, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar texts in Pinecone."""
        query_embedding = self.get_embeddings([query], client)[0]
        
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
