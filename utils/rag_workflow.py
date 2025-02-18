import asyncio
from typing import TypedDict, Dict, Any, List
from anthropic import Anthropic
import json
from utils.llm import LLMHandler
from utils.vector_store import VectorStore
from utils.tools import LuminosoStats

class State(TypedDict):
    query: str
    filters: Dict[str, Any]
    luminoso_results: Dict[str, Any]
    vector_results: List[Dict[str, Any]]
    final_response: str

async def generate_filters(state: State, llm_handler: LLMHandler) -> Dict[str, Any]:
    """Generate filters based on user query using Claude."""
    try:
        response = llm_handler.anthropic.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=2000,
            temperature=0,
            system="Extract metadata from the query and output valid JSON only.",
            messages=[{
                "role": "user",
                "content": f"""Given this query: {state['query']}

                Extract the following fields (leave empty if not specified):

                {{
                    "cities": [],
                    "rating_min": [],
                    "rating_max": [],
                    "month_start": [],
                    "year_start": [],
                    "month_end": [],
                    "year_end": [],
                    "location": [],
                    "states": [],
                    "themes": [],
                    "subsets": []
                }}

                Only use these valid values:
                - cities: Austin, Bellevue, Bethesda, Boston, Brooklyn, Chestnut Hill, Chicago, Denver, Houston, Los Angeles, Miami, Montreal, Nashville, New York, North York, Philadelphia, San Diego, Seattle, Short Hills, Skokie, Toronto, Vancouver, West Vancouver
                - rating_min/max: 1-5
                - month_start/end: 1-12
                - year_start/end: 2019-2025
                - states: NY, CA, TX, BC, MA, QC, ON, IL, WA, PA, MD, TN, FL, NJ, CO
                - themes: Exceptional Customer Service & Support, Poor Service & Long Wait Times, Product Durability & Quality Issues, Aesthetic Design & Visual Appeal, Professional Piercing Services & Environment, Store Ambiance & Try-On Experience, Price & Policy Transparency, Store Organization & Product Selection, Complex Returns & Warranty Handling, Communication & Policy Consistency, Value & Price-Quality Assessment, Affordable Luxury & Investment Value, Online Shopping Experience, Inventory & Cross-Channel Integration
                """
            }])

        filters = json.loads(response.content[0].text)
        return filters
    except Exception as e:
        print(f"Filter generation error: {str(e)}")
        return {}

async def get_luminoso_stats(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get statistics from Luminoso API based on filters."""
    try:
        luminoso_stats = LuminosoStats()
        client = luminoso_stats.initialize_client()

        drivers = luminoso_stats.fetch_drivers(client, state["filters"])
        sentiment = luminoso_stats.fetch_sentiment(client, state["filters"])

        return {
            "drivers": drivers.to_dict() if not drivers.empty else {},
            "sentiment": sentiment.to_dict() if not sentiment.empty else {}
        }
    except Exception as e:
        print(f"Luminoso stats error: {str(e)}")
        return {"drivers": {}, "sentiment": {}}

async def get_vector_results(state: Dict[str, Any], vector_store: VectorStore, llm_handler: LLMHandler) -> List[Dict[str, Any]]:
    """Get relevant reviews from vector store based on filters."""
    try:
        results = vector_store.filter_search(
            state["filters"],
            state['query'],
            llm_handler,
            top_k=5,
            index_name='reviews-csv-main'
        )
        return results
    except Exception as e:
        print(f"Vector search error: {str(e)}")
        return []

async def generate_final_response(state: Dict[str, Any], llm_handler: LLMHandler) -> str:
    """Generate final response combining all results."""
    try:
        response = llm_handler.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            system="You are a customer experience analysis expert. Provide insights from the data in a clear, structured format.",
            messages=[{
                "role": "user",
                "content": f"""
                Analyze this data and provide insights:

                User Query: {state['query']}

                Data Analysis:
                {json.dumps(state['luminoso_results'], indent=2)}

                Relevant Reviews:
                {json.dumps(state['vector_results'], indent=2)}

                Format your response with:
                1. Brief summary of the question
                2. Key insights (use bullet points)
                3. Data-backed findings
                4. Summary and recommendations
                """
            }])

        return response.content[0].text
    except Exception as e:
        print(f"Response generation error: {str(e)}")
        return "Error generating response. Please try again."

async def process_query(query: str, llm_handler: LLMHandler, vector_store: VectorStore) -> Dict[str, Any]:
    """Process a query through the workflow and return results."""
    try:
        # Initialize state
        state = {
            "query": query,
            "filters": {},
            "luminoso_results": {},
            "vector_results": [],
            "final_response": ""
        }

        # Step 1: Generate filters
        state["filters"] = await generate_filters(state, llm_handler)

        # Steps 2 & 3: Execute parallel tasks
        luminoso_task = asyncio.create_task(get_luminoso_stats(state))
        vector_task = asyncio.create_task(get_vector_results(state, vector_store, llm_handler))

        # Wait for both tasks to complete
        results = await asyncio.gather(luminoso_task, vector_task)

        # Update state with results
        state["luminoso_results"] = results[0]
        state["vector_results"] = results[1]

        # Step 4: Generate final response
        state["final_response"] = await generate_final_response(state, llm_handler)

        return state

    except Exception as e:
        print(f"Workflow error: {str(e)}")
        return {
            "query": query,
            "filters": {},
            "luminoso_results": {},
            "vector_results": [],
            "final_response": f"Error processing query: {str(e)}"
        }