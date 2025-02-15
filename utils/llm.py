import os
import sys
from anthropic import Anthropic
from typing import List, Dict, Any

class LLMHandler:
    def __init__(self):
        """Initialize Anthropic client."""
        # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        self.anthropic_key = (os.environ.get('ANTHROPIC_API_KEY') or
                            sys.exit('ANTHROPIC_API_KEY environment variable must be set'))
        
        self.client = Anthropic(api_key=self.anthropic_key)
        self.model = "claude-3-5-sonnet-20241022"

    def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate a response using the LLM."""
        context_text = "\n".join([f"- {c['text']}" for c in context])
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. "
                          "Use the context to provide accurate and relevant information."
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\n"
                          "Please provide a detailed answer based on the context provided."
            }
        ]
        
        response = self.client.messages.create(
            model=self.model,
            messages=messages
        )
        
        return response.content

    def get_client(self) -> Anthropic:
        """Return the Anthropic client."""
        return self.client
