from luminoso_api import LuminosoClient
from typing import Optional

def initialize_client() -> LuminosoClient:
    """
    Initialize and return a Luminoso API client instance.
    
    Args:
        api_url: The Luminoso API URL
        api_token: The API authentication token
    
    Returns:
        LuminosoClient: Initialized API client
    
    Raises:
        ValueError: If API credentials are missing
        Exception: For other initialization errors
    """
    try:
        client = LuminosoClient.connect(
            url='/projects/pr6j5grz/',
            token=''
        )
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize API client: {str(e)}")
