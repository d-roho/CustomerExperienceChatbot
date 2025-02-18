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
    vector_results: Dict[str, Any]
    final_response: str

async def generate_filters(state: State, llm_handler: LLMHandler) -> State:
    """Generate filters based on user query using Claude."""
    try:
        response = llm_handler.anthropic.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=2000,
            temperature=0,
            system="""Extract metadata from the query in the following format. 

        {'cities': [] ,
        'rating_min': [],
        'rating_max': [],
        'month_start': [],
        'year_start': [],
        'month_end': [],
        'year_end': [],
        'location': [],
        'states': [],
        'themes':  []
        'subsets': [] }

        Only choose from the possible values provided below. Leave the value blank if it is not specified in the query.  

        'cities': [Austin, Bellevue, Bethesda, Boston, Brooklyn, Chestnut Hill, Chicago, Denver, Houston, Los Angeles, Miami, Montreal, Nashville, New York, North York, Philadelphia, San Diego, Seattle, Short Hills, Skokie, Toronto, Vancouver, West Vancouver] ,
        'rating_min': [1,2,3,4,5],
        'rating_max': [1,2,3,4,5],
        'month_start': [1,2,3,4,5,6,7,8,9,10,11,12],
        'year_start': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'month_end':  [1,2,3,4,5,6,7,8,9,10,11,12],
        'year_end': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'location': [43 Spring St, New York, NY | 8404 Melrose Ave, Los Angeles, CA | 11700 Domain Blvd Suite 126, Austin, TX | 2166 W 4th Ave, Vancouver, BC, Canada | 126 Newbury St, Boston, MA | 1410 Peel St, Montreal, Quebec, Canada | 3401 Dufferin St, North York, ON, Canada | 940 W Randolph St, Chicago, IL, United States | 888 Westheimer Rd Suite 158, Houston, TX | 4545 La Jolla Village Dr Suite C-12, San Diego, CA | 2621 NE University Village St, Seattle, WA | 107 N 6th St, Brooklyn, NY | 144 5th Ave, New York, NY | 1525 Walnut St, Philadelphia, PA | 7247 Woodmont Ave, Bethesda, MD | 64 Ossington Ave, Toronto, ON, Canada | 2803 12th Ave S, Nashville, TN | 219 NW 25th St, Miami, FL | 925 Main St Unit H3, West Vancouver, BC, Canada | 124 Bellevue Square Unit L124, Bellevue, WA | 1200 Morris Tpke, Short Hills, NJ, United States | 3000 E 1st Ave #144, Denver, CO | 4999 Old Orchard Shopping Ctr Suite B34, Skokie, IL, United States | 737 Dunsmuir St, Vancouver, BC, Canada | 27 Boylston St, Chestnut Hill, MA]
        'states': [NY , CA , TX , BC , MA , QC , ON , IL , WA , PA , MD , TN , FL , NJ , CO],
        'themes':  [Exceptional Customer Service & Support , Poor Service & Long Wait Times , Product Durability & Quality Issues , Aesthetic Design & Visual Appeal , Professional Piercing Services & Environment, Store Ambiance & Try-On Experience , Price & Policy Transparency , Store Organization & Product Selection , Complex Returns & Warranty Handling , Communication & Policy Consistency , Value & Price-Quality Assessment , Affordable Luxury & Investment Value , Online Shopping Experience , Inventory & Cross-Channel Integration]
        'subsets': Go through the given query and find out all the fields that can be used for comparison with other subsets. These should be selected from the earlier used fields and should be highly relevant. You can give multiple fields if necessary. Use from these (cities, rating_min, rating_max, month_start, year_start, month_end, year_end, location, states, themes)
        """,
            messages=[{
                "role": "user",
                "content": f"Query: {state['query']}"
            }])
        state["filters"] = json.loads(response.content[0].text)
        return state
    except Exception as e:
        raise RuntimeError(f"Filter generation failed: {str(e)}")

async def get_luminoso_stats(state: State) -> State:
    """Get statistics from Luminoso API based on filters."""
    try:
        luminoso_stats = LuminosoStats()
        client = luminoso_stats.initialize_client()

        # Get both drivers and sentiment analysis
        drivers = luminoso_stats.fetch_drivers(client, state["filters"])
        sentiment = luminoso_stats.fetch_sentiment(client, state["filters"])

        state["luminoso_results"] = {
            "drivers": drivers.to_dict(),
            "sentiment": sentiment.to_dict()
        }
        return state
    except Exception as e:
        raise RuntimeError(f"Luminoso stats retrieval failed: {str(e)}")

async def get_vector_results(state: State, vector_store: VectorStore, llm_handler: LLMHandler) -> State:
    """Get relevant reviews from vector store based on filters."""
    try:
        results = vector_store.filter_search(
            state["filters"],
            state['query'],
            llm_handler,
            top_k=5,  # Default value, can be made configurable
            index_name='reviews-csv-main'
        )
        state["vector_results"] = results
        return state
    except Exception as e:
        raise RuntimeError(f"Vector store search failed: {str(e)}")

async def generate_final_response(state: State, llm_handler: LLMHandler) -> State:
    """Generate final response combining all results."""
    try:
        response = llm_handler.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            system="""You are a helpful customer experience analysis expert that provides insights from aggregate ratings and sentiment data and customer reviews.

            Provide a well-structured response that:
            Intro:
            A basic line to summarize the question and give a start to the answer. 
            Keep it neutral and informative
            Body: 
            Bullet points should be used wherever necessary
            Reference the source wherever using them to answer the question
            Use numbers and percentages when necessary
            End: 
            Summary of the answer in 2-3 lines emphasizing the most important points.
            Suggested actions based on findings if necessary
            Offer to help with any other question
            """,
            messages=[{
                "role": "user",
                "content": f"""
                User Query: {state['query']}

                Relevant Data Analysis:
                {json.dumps(state['luminoso_results'], indent=2)}

                Relevant Reviews:
                {json.dumps(state['vector_results'], indent=2)}
                """
            }])

        state["final_response"] = response.content[0].text
        return state
    except Exception as e:
        raise RuntimeError(f"Final response generation failed: {str(e)}")

async def process_query(query: str, llm_handler: LLMHandler, vector_store: VectorStore) -> Dict:
    """Process a query through the workflow and return the final response."""
    try:
        # Initialize state
        state = State(
            query=query,
            filters={},
            luminoso_results={},
            vector_results={},
            final_response=""
        )

        # Step 1: Generate filters
        state = await generate_filters(state, llm_handler)

        # Steps 2 & 3: Parallel execution of Luminoso stats and vector search
        luminoso_task = asyncio.create_task(get_luminoso_stats(state))
        vector_task = asyncio.create_task(get_vector_results(state, vector_store, llm_handler))

        # Wait for both tasks to complete
        results = await asyncio.gather(luminoso_task, vector_task)

        # Merge results back into state
        state["luminoso_results"] = results[0]["luminoso_results"]
        state["vector_results"] = results[1]["vector_results"]

        # Step 4: Generate final response
        state = await generate_final_response(state, llm_handler)

        return state

    except Exception as e:
        raise RuntimeError(f"Query processing failed: {str(e)}")