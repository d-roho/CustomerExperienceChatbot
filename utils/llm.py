import os
import sys
from anthropic import Anthropic
from typing import List, Dict, Any


class LLMHandler:

    def __init__(self):
        """Initialize Anthropic client."""
        # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        self.anthropic_key = (
            os.environ.get('ANTHROPIC_API_KEY')
            or sys.exit('ANTHROPIC_API_KEY environment variable must be set'))

        self.client = Anthropic(api_key=self.anthropic_key)
        self.model = "claude-3-5-sonnet-20241022"

    def generate_response(self, query: str, context: List[Dict[str,
                                                               Any]]) -> str:
        """Generate a response using the LLM."""
        context_text = "\n".join([f"- {c['text']}" for c in context])

        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"""Context information is below.
                        ---------------------
                        {context_text}
                        ---------------------
                        Given the context information, answer the following question: {query}
                        Answer:"""
                }],
                max_tokens=1000,
                temperature=0.0)

            return response.content[0].text

        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using Claude."""
        try:
            embeddings = []
            for text in texts:
                response = self.client.embeddings.create(
                    model="claude-3-5-sonnet-20241022", input=text)
                embeddings.append(response.embeddings[0])
            return embeddings
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

    def get_client(self) -> Anthropic:
        """Return the Anthropic client."""
        return self.client
