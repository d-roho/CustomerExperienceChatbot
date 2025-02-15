import nltk
from typing import List
import re

class TextProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    def clean_text(self, text: str) -> str:
        """Clean the input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        # Clean the text first
        text = self.clean_text(text)
        
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
                # Add the current chunk to chunks
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_size = 0
                current_chunk = []
                
                # Add sentences from the previous chunk for overlap
                while overlap_size < self.chunk_overlap and current_chunk:
                    last_sentence = current_chunk.pop()
                    overlap_size += len(last_sentence)
                    
                current_chunk = [sentence]
                current_length = sentence_length
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def process_file(self, file_content: str) -> List[str]:
        """Process an uploaded file content."""
        return self.chunk_text(file_content)
