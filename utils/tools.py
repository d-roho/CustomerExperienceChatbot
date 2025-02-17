import os
from typing import Dict, Any, List, Optional
import json
from luminoso_api import LuminosoClient


class LuminosoStats:

    def __init__(self):
        """Initialize LUMINOSO API connection."""
        self.luminoso_token = os.environ.get('LUMINOSO_TOKEN')
        if not self.luminoso_token:
            raise ValueError("LUMINOSO_TOKEN environment variable is required")
        self.luminoso_url = os.environ.get('LUMINOSO_URL')
        if not self.luminoso_url:
            raise ValueError("LUMINOSO_URL environment variable is required")

    def initialize_client(self) -> LuminosoClient:
        """     
        Returns:
            LuminosoClient: Initialized API client

        Raises:
            ValueError: If API credentials are missing
            Exception: For other initialization errors
        """
        try:
            client = LuminosoClient.connect(url=self.luminoso_url,
                                            token=self.luminoso_token)
            return client
        except Exception as e:
            raise Exception(
                f"Failed to initialize Luminoso API client: {str(e)}")

    def fetch_drivers(self, client):
        """Fetch drivers for given filters and concepts."""
        try:
            concept = {
                "type":
                "concept_list",
                'name':
                'Shopping Experience'
            }
            filter = [{
                   "name": "Overall Rating",
                   "maximum": 5
               }]
            result = client.get('/concepts/score_drivers/', score_field="Overall Rating", concept_selector=concept, filter=filter
                            )

            df = pd.DataFrame(result)
            df = df.drop(columns=['shared_concept_id', 'color', 'texts', 'exact_term_ids', 'excluded_term_ids', 'vectors', 'exact_match_count'])
            # Select numeric columns (excluding 'name')
            numeric_cols = df.select_dtypes(include=['number']).columns
    
            # Normalize each numeric column
            for col in numeric_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if col in ['average_score', 'baseline']:
                # Handle constant columns by setting to 0.5
                df[col] = df[col]/5.0
            elif col == 'impact':
                df[col] = df[col]
            else:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    
            column_mapping = {
            'name': 'Theme Name',
            'relevance': 'Normalized Relevance to Subset',
            'match_count': 'Matches Found in Subset',
            'impact': 'Impact on Score',
            'confidence': 'Impact Confidence Level',
            'average_score': 'Average Score',
            'baseline': 'Baseline Score'
            }
    
            # Rename the columns using the mapping
            df = df.rename(columns=column_mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to Fetch Drivers: {str(e)}")

    def fetch_sentiment(self, client):
        """Fetch sentiment for given filters and concepts."""
        try:
            concept = {
                "type":
                "concept_list",
                'name':
                'Shopping Experience'
            }
            filter = [{
                   "name": "Overall Rating",
                   "maximum": 5
               }]
            result = client.get('/concepts/sentiment/', concept_selector=concept, filter=filter
                            )

            rows = []
            for concept in result['match_counts']:
                row = {
                    'Theme Name': concept['name'],
                    'Proportion of Subset With Theme': concept['match_count'],
                    'Proportion of Positive Mentions': concept['sentiment_share']['positive'],
                    'Proportion of Neutral Mentions': concept['sentiment_share']['neutral'],
                    'Proportion of Negative Mentions': concept['sentiment_share']['negative']
                }
                rows.append(row)
                filter_count = result['filter_count'] 
    
            # Create DataFrame
            df = pd.DataFrame(rows)
            df['Proportion of Subset With Theme'] = df['Proportion of Subset With Theme']/filter_count 
        except Exception as e:
            raise RuntimeError(f"Failed to Fetch Drivers: {str(e)}")



