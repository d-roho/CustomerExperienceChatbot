import re
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from anthropic import Anthropic
# from sentence_transformers import SentenceTransformer
# import torch
from utils.db import MotherDuckStore  # New import
import datetime



class VectorStore:

    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initialize Pinecone vector store and MotherDuck store."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.db = MotherDuckStore()  # Initialize MotherDuck store

        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=api_key)
            print(
                f"Pinecone initialized. Available indexes: {self.pc.list_indexes().names()}"
            )

            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating new index: {self.index_name}")
                try:
                    self.pc.create_index(name=self.index_name,
                                         dimension=self.dimension,
                                         metric='cosine',
                                         spec=ServerlessSpec(
                                             cloud='aws',
                                             region=self.environment))
                    print(f"Successfully created index: {self.index_name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to create Pinecone index: {str(e)}")

            self.index = self.pc.Index(self.index_name)
            print(f"Successfully connected to index: {self.index_name}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone: {str(e)}")

    def upsert_texts(self, chunks: List[Dict[str, Any]], client: Anthropic,
                     index_name, df) -> None:
        """Upload text chunks with metadata to Pinecone and MotherDuck."""
        try:
            print(f"\nStarting embedding process for {len(chunks)} chunks")

            # Extract texts for embedding
            texts = [chunk['text'] for chunk in chunks]
            print("Average chunk size:",
                  sum(len(t) for t in texts) / len(texts), "characters")
            print("Largest chunk size:", max(len(t) for t in texts),
                  "characters")

            print("\nGenerating embeddings...")
            embeddings = client.get_embeddings(texts)
            print(f"Successfully generated {len(embeddings)} embeddings")

            # Store texts in MotherDuck DB
            print("\nStoring texts in MotherDuck DB...")
            self.db.create_table(index_name, df)
            print("Successfully stored all chunks in MotherDuck")

            # Prepare vectors with metadata
            print("\nPreparing vectors for Pinecone upload...")
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vectors.append(
                    (chunk['metadata']['id'], embedding, chunk['metadata']))

            # Batch upsert to Pinecone
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            print(
                f"\nStarting batch upload process ({total_batches} batches)..."
            )

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                current_batch = i // batch_size + 1
                print(
                    f"\nUpserting batch {current_batch}/{total_batches} ({len(batch)} vectors)"
                )
                self.index.upsert(vectors=batch)
                print(
                    f"Batch {current_batch} complete - {i + len(batch)}/{len(vectors)} vectors processed"
                )

            print("\nVector upload complete!")
            print(
                f"Successfully processed {len(vectors)} vectors across {total_batches} batches"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to upsert texts: {str(e)}")

    def search(self,
               query: str,
               client: Anthropic,
               top_k: int = 5,
               index_name: str = 'reviews-csv-main') -> List[Dict[str, Any]]:
        """Search for similar texts using Pinecone and retrieve full texts from MotherDuck."""

        try:
            print(f"Getting embedding for query: {query[:50]}...")
            query_embedding = client.get_embeddings([query])[0]
            print(f"Searching Pinecone with top_k={top_k}")
            results = self.index.query(vector=query_embedding,
                                       top_k=top_k,
                                       include_metadata=True)

            print(f"Successfully retrieved {len(results['matches'])} results")
            # Retrieve full texts and metadata from MotherDuck
            processed_results = []
            for match in results.matches:
                print(match)
                stored_data = self.db.get_chunk(match.id, index_name)
                if stored_data:
                    processed_results.append({
                        'text':
                        stored_data['text'],
                        'metadata':
                        stored_data['metadata'],
                        'header':
                        match.metadata['header'],
                        'score':
                        match.score
                    })

            return processed_results

        except Exception as e:
            raise RuntimeError(f"Failed to search: {str(e)}")

    def filter_search(
        self,
        filters: Dict[str, Any],
        query: str,
        client: Anthropic,
        top_k: int = 5,
        index_name: str = 'reviews-csv-main',
    ) -> List[Dict[str, Any]]:
        """Search for similar texts using METADATA FILTERS."""

        try:
            print(f"Getting embedding for query: {query[:50]}...")
            query_embedding = client.get_embeddings([query])[0]
            print(f"Searching Pinecone with top_k={top_k}")

            FIELD_MAPPING = {
                'cities': 'city',
                'states': 'state',
                'month_start': 'date_month',
                'year_start': 'date_year',
                'month_end': 'date_month',
                'year_end': 'date_year',
                # 'themes': 'themes',
                'rating_min': 'rating',
                'rating_max': 'rating',
                'location': 'location'
                # subset of fields to search for
            }
            # SCALAR_FIELDS = {'month_start', 'year_start', 'month_end', 'year_end', 'rating_min', 'rating_min'}

            filter_query = {}

            # Handle special rating range case
            rating_conditions = {}
            if filters.get('rating_min'):
                rating_conditions['$gte'] = filters['rating_min'][0]
            if filters.get('rating_max'):
                rating_conditions['$lte'] = filters['rating_max'][0]
            if rating_conditions:
                filter_query['rating'] = rating_conditions

            # Handle special date range case
            def get_unix_time(month, year):
                # Create a datetime object for the first day of the given month and year
                dt = datetime.datetime(year, month, 1, 0, 0, 0)

                # Convert the datetime object to Unix time (seconds since January 1, 1970)
                unix_time = int(dt.timestamp())
                return unix_time
          
            
            date_conditions = {}

            
            if filters.get('year_start'):
                if filters.get('month_start'):
                    start_unix = get_unix_time(filters['month_start'][0], filters['year_start'][0])
                else:
                    start_unix = get_unix_time(1, filters['year_start'][0])
                date_conditions['$gte'] = start_unix

            if filters.get('year_end'):
                if filters.get('month_end'):
                    end_unix = get_unix_time(filters['month_end'][0], filters['year_end'][0])
                else:
                    end_unix = get_unix_time(12, filters['year_end'][0])
                date_conditions['$lte'] = end_unix

            if date_conditions:
                filter_query['date_unix'] = date_conditions

            # Build filter query
            for key in filters:
                if key in ['rating_min', 'rating_max', 'subsets', 'month_start', 'year_start',
                    'month_end','year_end']:
                    continue  # Already handled or will handle later

                values = filters[key]
                if not values:
                    continue  # Skip empty lists

                if key in FIELD_MAPPING:
                    mongo_field = FIELD_MAPPING[key]

                    if key.lower() == 'themes':
                        for theme in filters[key]:  
                            filter_query[theme] = '{"$exists": true}'
                    else:
                        # Handle array values with $in
                        filter_query[mongo_field] = {'$in': values}

            # # Build projection
            # projection = {}
            # if 'subsets' in filters:
            #     for field in filters['subsets']:
            #         if field in FIELD_MAPPING:
            #             projection[FIELD_MAPPING[field]] = 1
            #     if filters['subsets']:  # Only exclude _id if we have subsets
            #         projection['_id'] = 0

            print(filter_query)
            
            results = self.index.query(vector=query_embedding,
                                       top_k=top_k,
                                       filter=filter_query,
                                       include_metadata=True)

            print(f"Successfully retrieved {len(results['matches'])} results")
            # Retrieve full texts and metadata from MotherDuck
            processed_results = []
            for match in results.matches:
                print(match)
                stored_data = self.db.get_chunk(match.id, index_name)
                if stored_data:
                    processed_results.append({
                        'text':
                        stored_data['text'],
                        'metadata':
                        stored_data['metadata'],
                        'header':
                        match.metadata['header'],
                        'score':
                        match.score
                    })

            return processed_results

        except Exception as e:
            raise RuntimeError(f"Failed to filter search: {str(e)}")

    def rerank_results(self, query: str,
                       results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank results using semantic similarity."""
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            print(f"Warning: Reranking model not available: {str(e)}")
            return results

        try:
            # Prepare pairs for reranking
            pairs = [[query, f"{result['header']}\n{result['text']}"]
                     for result in results]
            scores = model.predict(pairs)

            # Update scores
            for i, result in enumerate(results):
                result['score'] = float(scores[i])

            # Sort by new scores
            results.sort(key=lambda x: x['score'], reverse=True)
            return results
        except Exception as e:
            print(f"Warning: Reranking failed: {str(e)}")
            return results
