import os
import duckdb
from typing import Dict, Any, List
import json


class MotherDuckStore:

    def __init__(self):
        """Initialize MotherDuck connection."""
        motherduck_token = os.environ.get('MOTHERDUCK_TOKEN')
        if not motherduck_token:
            raise ValueError(
                "MOTHERDUCK_TOKEN environment variable is required")

        self.conn_str = f"md:reviews?token={motherduck_token}"
        self.conn = duckdb.connect(self.conn_str)

        # Create text_chunks table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS text_chunks (
                id VARCHAR PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSON
            )
        """)
        self.conn.commit()

    def create_table(self, index_name: str, df):
        """Create tables using a given pandas dataframe."""
        try:
            # Clean column names: replace spaces and special chars with underscores
            df.columns = [
                col.replace(' ', '_').replace('-', '_') for col in df.columns
            ]

            # Create a temp view of the dataframe
            self.conn.register('temp_df', df)

            self.conn.execute(f"""
                DROP TABLE IF EXISTS "{index_name}"
                """)

            # Create the table from the temp view
            self.conn.execute(f"""
                CREATE TABLE "{index_name}" AS 
                SELECT * FROM temp_df
            """)
            self.conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to create table: {str(e)}")

    def store_chunk(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        """Store a text chunk with its metadata."""
        try:
            metadata_json = json.dumps(metadata)
            self.conn.execute(
                """
                INSERT INTO text_chunks (id, text, metadata)
                VALUES (?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
            """, [chunk_id, text, metadata_json])
            self.conn.commit()
        except Exception as e:
            raise RuntimeError(f"Failed to store chunk: {str(e)}")

    def store_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """Store multiple chunks in a batch."""
        try:
            for chunk in chunks:
                self.store_chunk(chunk['metadata']['id'], chunk['text'],
                                 chunk['metadata'])
        except Exception as e:
            raise RuntimeError(f"Failed to store chunks batch: {str(e)}")

    def get_chunk(self, chunk_id: str, index_name: str) -> Dict[str, Any]:
        """Retrieve a chunk by its ID."""
        try:
            run = f"""
                SELECT * 
                FROM "{index_name}"
                WHERE uuid = '{chunk_id}'
            """
            df = self.conn.execute(run).df()
            text = "none"
            metadata = "none"
            if not df.empty:
                text = df.iloc[0]['Text']
                metadata = df.iloc[0, 1:].to_dict()
            return {'text': text, 'metadata': metadata}
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve chunk: {str(e)}")

    def fetch_all(self, index_name: str):
        """Retrieve the whole index table"""
        try:
            run = f"""
                SELECT * 
                FROM "{index_name}"
            """
            print(run)
            df = self.conn.execute(run).df()
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve table: {str(e)}")

