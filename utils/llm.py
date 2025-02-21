import os
import sys
from anthropic import Anthropic
from openai import OpenAI
from typing import List, Dict, Any
import json


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
        self.model = None  # Will be set during generation

    def generate_response(self, query: str, context: List[Dict[str, Any]],
                          model: str, max_tokens: int = 2000) -> str:
        """Generate a response using Claude."""

        if isinstance(context, dict):
            context_list = []
            cumulative_reviews = []
            for key in context.keys():
                value = context[key]
                title = f"Subset {key} Info: \n {json.dumps(value['subset_info'], indent=2)}"

                context_text = "\n".join(
                    [
                        f"Review {idx} (Retriever Score: {c['score']}) \nMetadata: {c['header']} \n - Text: {c['text']}\n\n"
                        if c['header'] not in cumulative_reviews
                        else f"Review {idx} \n {c['text']}"
                        for idx, c in enumerate(value['processed_results'])
                    ]
                )

                
                context_list.append(f"{title}\n\n {context_text} \n\n END OF SUBSET {key} \n\n")    
                cumulative_reviews.extend([c['header'] for idx, c in enumerate(value['processed_results'])]) #remove duplicate reviews       
            context_text = "\n\n".join(context_list)

            
                         
        else:
            context_text = "\n".join([
                f" Review {idx} (Retriever Score: {c['score']}) \nMetadata: {c['header']} \n - Text: {c['text']}\n\n"
                for idx, c in enumerate(context)
            ])
    
        try:
            response = self.anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                system=
                "You are a knowledgeable assistant. Use the provided context to answer questions accurately.",
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"Context:\n{context_text}\n\nQuestion: {query}"
                }])
            return response.content[0].text, context_text
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
