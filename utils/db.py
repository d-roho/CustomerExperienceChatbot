import os
import duckdb
from typing import Dict, Any, List
import json
import pandas as pd

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
        # Create reviews table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id VARCHAR PRIMARY KEY,
                review_text TEXT NOT NULL,
                city VARCHAR,
                helpful_count INTEGER,
                created_date TIMESTAMP,
                store_name VARCHAR,
                rating FLOAT,
                location VARCHAR,
                state VARCHAR,
                user_name VARCHAR,
                concepts TEXT,
                embedding_id VARCHAR UNIQUE
            )
        """)

        # Create text_chunks table for RAG
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id VARCHAR PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSON,
                review_id VARCHAR,
                FOREIGN KEY (review_id) REFERENCES reviews(id)
            )
        """)
        self.conn.commit()

    def import_reviews_from_csv(self, csv_path: str):
        """Import reviews from CSV file."""
        try:
            # Read CSV using pandas
            df = pd.read_csv(csv_path)

            # Generate UUIDs for each review
            df['id'] = [str(i) for i in range(len(df))]

            # Rename columns to match table schema
            df = df.rename(columns={
                'Text': 'review_text',
                'string_City': 'city',
                'score_Count People Found Review Helpful': 'helpful_count',
                'date_Date Created': 'created_date',
                'string_Name': 'store_name',
                'score_Overall Rating': 'rating',
                'string_Place Location': 'location',
                'string_State': 'state',
                'string_User Name': 'user_name',
                'Concepts': 'concepts'
            })

            # Add empty embedding_id column
            df['embedding_id'] = None

            # Insert data into reviews table
            self.conn.execute("""
                INSERT INTO reviews 
                SELECT * FROM df
                ON CONFLICT (id) DO NOTHING
            """)
            self.conn.commit()

            return f"Successfully imported {len(df)} reviews"

        except Exception as e:
            raise RuntimeError(f"Failed to import reviews: {str(e)}")

    def store_chunk(self, chunk_id: str, text: str, metadata: Dict[str, Any], review_id: str = None):
        """Store a text chunk with its metadata."""
        try:
            metadata_json = json.dumps(metadata)
            self.conn.execute("""
                INSERT INTO text_chunks (id, text, metadata, review_id)
                VALUES (?, ?, ?, ?)
                """, [chunk_id, text, metadata_json, review_id])
            self.conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to store chunk: {str(e)}")

    def store_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Store multiple chunks in a batch."""
        try:
            for chunk in chunks:
                review_id = chunk['metadata'].get('review_id')
                self.store_chunk(
                    chunk['metadata']['id'],
                    chunk['text'],
                    chunk['metadata'],
                    review_id
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

    def get_review(self, review_id: str) -> Dict[str, Any]:
        """Retrieve a review by its ID."""
        try:
            result = self.conn.execute("""
                SELECT *
                FROM reviews
                WHERE id = ?
            """, [review_id]).fetchone()

            if result:
                columns = ['id', 'review_text', 'city', 'helpful_count', 'created_date', 
                          'store_name', 'rating', 'location', 'state', 'user_name', 
                          'concepts', 'embedding_id']
                return dict(zip(columns, result))
            return None
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve review: {str(e)}")