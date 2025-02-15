import nltk
from typing import List, Dict, Any
import re
import pandas as pd
import uuid

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def clean_text(self, text: str) -> str:
        """Clean the input text."""
        if not text or pd.isna(text):
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text)).strip()
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def process_csv_file(self, file_content: bytes) -> List[Dict[str, Any]]:
        """Process CSV file and return chunks with metadata."""
        try:
            # Read CSV from bytes using BytesIO
            import io
            df = pd.read_csv(io.BytesIO(file_content))

            # Ensure 'Text' column exists
            if 'Text' not in df.columns:
                raise ValueError("CSV must contain a 'Text' column")

            # Process each row
            chunks = []
            for _, row in df.iterrows():
                # Clean the review text
                text = self.clean_text(row['Text'])
                if not text:  # Skip empty texts
                    continue

                # Create metadata dictionary
                metadata = {
                    'id': str(uuid.uuid4()),
                    'city': row.get('string_City', ''),
                    'rating': row.get('score_Overall Rating', 0),
                    'date': row.get('date_Date Created', ''),
                    'location': row.get('string_Place Location', '')
                }

                chunks.append({
                    'text': text,
                    'metadata': metadata
                })

            return chunks

        except Exception as e:
            raise RuntimeError(f"Failed to process CSV file: {str(e)}")

    def process_text_file(self, file_content: str) -> List[Dict[str, Any]]:
        """Process plain text file and return chunks."""
        # Clean the text
        text = self.clean_text(file_content)

        # Split into sentences
        sentences = nltk.sent_tokenize(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Add the current chunk to chunks if it exists
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'metadata': {
                            'id': str(uuid.uuid4())
                        }
                    })

                # Start new chunk
                current_chunk = [sentence]
                current_length = sentence_length

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'metadata': {
                    'id': str(uuid.uuid4())
                }
            })

        return chunks

    def process_file(self, file_content: bytes, file_type: str) -> List[Dict[str, Any]]:
        """Process an uploaded file based on its type."""
        if file_type == 'csv':
            return self.process_csv_file(file_content)
        elif file_type == 'txt':
            return self.process_text_file(file_content.decode())
        else:
            raise ValueError(f"Unsupported file type: {file_type}")