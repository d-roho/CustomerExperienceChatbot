import os
import duckdb
from typing import Dict, Any, List
import json

class MotherDuckStore:
    def __init__(self):
        """Initialize MotherDuck connection."""
        motherduck_token = os.environ.get('MOTHERDUCK_TOKEN')
        if not motherduck_token:
            raise ValueError("MOTHERDUCK_TOKEN environment variable is required")
        
        self.conn_str = f"md:reviews?token={motherduck_token}"
        self.conn = duckdb.connect(self.conn_str)
        self._initialize_tables()

    def _initialize_tables(self):
        """Create tables if they don't exist."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id VARCHAR PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSON
            )
        """)
        self.conn.commit()

    def store_chunk(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        """Store a text chunk with its metadata."""
        try:
            metadata_json = json.dumps(metadata)
            self.conn.execute("""
                INSERT INTO text_chunks (id, text, metadata)
                VALUES (?, ?, ?)
                """, [chunk_id, text, metadata_json])
            self.conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to store chunk: {str(e)}")

    def store_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Store multiple chunks in a batch."""
        try:
            for chunk in chunks:
                self.store_chunk(
                    chunk['metadata']['id'],
                    chunk['text'],
                    chunk['metadata']
                )
        except Exception as e:
            raise RuntimeError(f"Failed to store chunks batch: {str(e)}")

    def get_chunk(self, chunk_id: str) -> Dict[str, Any]:
        """Retrieve a chunk by its ID."""
        try:
            result = self.conn.execute("""
                SELECT text, metadata
                FROM text_chunks
                WHERE id = ?
            """, [chunk_id]).fetchone()
            
            if result:
                text, metadata_json = result
                return {
                    'text': text,
                    'metadata': json.loads(metadata_json)
                }
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve chunk: {str(e)}")
