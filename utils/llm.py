import os
import sys
from anthropic import Anthropic
from openai import OpenAI
from typing import List, Dict, Any


class LLMHandler:

    def __init__(self):
        """Initialize Anthropic and OpenAI clients."""
        self.anthropic_key = (
            os.environ.get('ANTHROPIC_API_KEY')
            or sys.exit('ANTHROPIC_API_KEY environment variable must be set'))

        self.openai_key = (
            os.environ.get('OPENAI_API_KEY')
            or sys.exit('OPENAI_API_KEY environment variable must be set'))

        self.anthropic = Anthropic(api_key=self.anthropic_key)
        self.openai = OpenAI(api_key=self.openai_key)
        self.model = "claude-3-5-sonnet-20241022"

    def generate_response(self, query: str, context: List[Dict[str,
                                                               Any]]) -> str:
        """Generate a response using Claude."""
        context_text = "\n".join([f"- {c['text']}" for c in context])

        try:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0,
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are a knowledgeable assistant. Use the provided context to answer questions accurately."
                }, {
                    "role":
                    "user",
                    "content":
                    f"Context:\n{context_text}\n\nQuestion: {query}"
                }])
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI's ada-002."""
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-ada-002", input=texts)
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def get_client(self) -> Anthropic:
        """Return the Anthropic client."""
        return self.anthropic
