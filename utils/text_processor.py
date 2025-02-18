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
            df['uuid'] = [
                str(uuid.uuid4()).replace("-", "")
                for _ in range(len(df.index))
            ]

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
                # date
                timestamp = pd.to_datetime(row['date_Date Created'])
                month = timestamp.month  # No .dt needed for single Timestamp
                year = timestamp.year    # No .dt needed for single Timestamp
                date_unix = timestamp.timestamp()  # Directly get Unix timestamp


                master_themes = [
                    'Exceptional Customer Service & Support',
                    'Poor Service & Long Wait Times',
                    'Product Durability & Quality Issues',
                    'Aesthetic Design & Visual Appeal',
                    'Professional Piercing Services & Environment',
                    'Piercing Complications & Jewelry Quality',
                    'Store Ambiance & Try-On Experience',
                    'Price & Policy Transparency',
                    'Store Organization & Product Selection',
                    'Complex Returns & Warranty Handling',
                    'Communication & Policy Consistency',
                    'Value & Price-Quality Assessment',
                    'Affordable Luxury & Investment Value',
                    'Online Shopping Experience',
                    'Inventory & Cross-Channel Integration'
                ]
                row_themes = row['themes']

                # Create metadata dictionary
                # Helper functions for safe value extraction
                def get_str(row, key, default=''):
                    """Safely extract and convert string values from row"""
                    val = row.get(key)
                    return str(val) if pd.notna(val) else default

                def get_float(row, key, default=0.0):
                    """Safely extract and convert float values from row"""
                    val = row.get(key)
                    return float(val) if pd.notna(val) else default

                # Create metadata dictionary
                metadata = {
                    'id': row['uuid'],  # Assuming uuid is mandatory
                    'city': get_str(row, 'string_City'),
                    'rating': get_float(row, 'score_Overall Rating'),
                    'date_unix': date_unix,
                    'date_month': month,
                    'date_year': year,
                    'location': get_str(row, 'string_Place Location'),
                    'likes': get_float(row, 'score_Count People Found Review Helpful'),
                    'state': get_str(row, 'string_State'),
                }
                
                for theme in master_themes:
                    if theme in row_themes:
                        metadata[theme] = True
                        
                metadata[
                    'header'] = f"Location - {metadata['location']}, Date (MM/YYYY) - {metadata['date_month']}/{metadata['date_year']}, Rating - {metadata['rating']}/5.0, Upvotes {metadata['likes']}\nThemes - {row_themes}"
                
                # fulltext = f"{metadata['header']}\n Review - {text}"
                chunks.append({'text': text, 'metadata': metadata})
            return chunks, df

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

    def process_file(self, file_content: bytes,
                     file_type: str) -> List[Dict[str, Any]]:
        """Process an uploaded file based on its type."""
        if file_type == 'csv':
            return self.process_csv_file(file_content)
        elif file_type == 'txt':
            return self.process_text_file(file_content.decode())
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
