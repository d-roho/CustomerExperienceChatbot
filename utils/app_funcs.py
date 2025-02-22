import streamlit as st
import os
import time  # Added import for time module
from typing import Optional, Tuple, List, Dict, Any
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.llm import LLMHandler
import pandas as pd
from pinecone import ServerlessSpec
import json
from utils.rag_workflow import process_query, process_query_lite
from utils.theme_workflow import process_themes
import asyncio
from datetime import datetime
from utils.tools import LuminosoStats
import nest_asyncio
nest_asyncio.apply()


def init_components() -> Tuple[LLMHandler, VectorStore]:
  """Initialize LLM and Vector Store components with error handling."""
  try:
      llm_handler = LLMHandler()

      # Get environment variables with proper error handling
      api_key: Optional[str] = os.environ.get('PINECONE_API_KEY')
      environment: Optional[str] = os.environ.get('PINECONE_ENVIRONMENT')

      if not api_key or not environment:
          st.error(
              "Missing required Pinecone credentials. Please check your environment variables."
          )
          st.stop()

      try:
          vector_store = VectorStore(api_key=api_key,
                                     environment=environment,
                                     index_name='reviews-csv-main')
          return llm_handler, vector_store
      except Exception as e:
          st.error(f"Failed to initialize Pinecone: {str(e)}")
          st.stop()

  except Exception as e:
      st.error(f"Failed to initialize components: {str(e)}")
      st.stop()




# Variables

cities_list = [
    "Austin", "Bellevue", "Bethesda", "Boston", "Brooklyn",
    "Chestnut Hill", "Chicago", "Denver", "Houston", "Los Angeles",
    "Miami", "Montreal", "Nashville", "New York", "North York",
    "Philadelphia", "San Diego", "Seattle", "Short Hills", "Skokie",
    "Toronto", "Vancouver", "West Vancouver"
]
states_list = [
    "NY", "CA", "TX", "BC", "MA", "QC", "ON", "IL", "WA", "PA", "MD", "TN",
    "FL", "NJ", "CO"
]


locations_list = [
    "43 Spring St, New York, NY", "8404 Melrose Ave, Los Angeles, CA",
    "11700 Domain Blvd Suite 126, Austin, TX",
    "2166 W 4th Ave, Vancouver, BC, Canada", "126 Newbury St, Boston, MA",
    "1410 Peel St, Montreal, Quebec, Canada",
    "3401 Dufferin St, North York, ON, Canada",
    "940 W Randolph St, Chicago, IL, United States",
    "888 Westheimer Rd Suite 158, Houston, TX",
    "4545 La Jolla Village Dr Suite C-12, San Diego, CA",
    "2621 NE University Village St, Seattle, WA",
    "107 N 6th St, Brooklyn, NY", "144 5th Ave, New York, NY",
    "1525 Walnut St, Philadelphia, PA", "7247 Woodmont Ave, Bethesda, MD",
    "64 Ossington Ave, Toronto, ON, Canada",
    "2803 12th Ave S, Nashville, TN", "219 NW 25th St, Miami, FL",
    "925 Main St Unit H3, West Vancouver, BC, Canada",
    "124 Bellevue Square Unit L124, Bellevue, WA",
    "1200 Morris Tpke, Short Hills, NJ, United States",
    "3000 E 1st Ave #144, Denver, CO",
    "4999 Old Orchard Shopping Ctr Suite B34, Skokie, IL, United States",
    "737 Dunsmuir St, Vancouver, BC, Canada",
    "27 Boylston St, Chestnut Hill, MA"
]

themes_list = [
    "Exceptional Customer Service & Support",
    "Poor Service & Long Wait Times",
    "Product Durability & Quality Issues",
    "Aesthetic Design & Visual Appeal",
    "Professional Piercing Services & Environment",
    "Store Ambiance & Try-On Experience", "Price & Policy Transparency",
    "Store Organization & Product Selection",
    "Complex Returns & Warranty Handling",
    "Communication & Policy Consistency",
    "Value & Price-Quality Assessment",
    "Affordable Luxury & Investment Value", "Online Shopping Experience",
    "Inventory & Cross-Channel Integration"
]
