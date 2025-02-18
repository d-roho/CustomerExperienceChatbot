import langgraph.graph as lg
from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, Annotated, Sequence, Dict, Any
import json
from .llm import LLMHandler
from .tools import LuminosoStats
import asyncio
from typing import List, Dict, Any

class State(TypedDict):
    query: str
    filters: Dict[str, Any]
    luminoso_results: Dict[str, Any]
    vector_results: Dict[str, Any]
    final_response: str

async def generate_filters(state: State, llm_handler: LLMHandler) -> State:
    """Generate filters based on user query using Claude."""
    try:
        response, _ = llm_handler.generate_response(
            state["query"],
            [],  # No context needed for filter generation
            "claude-3-5-haiku-20241022",
            max_tokens=1000,
            system_prompt="""You are a helpful assistant that generates search filters for analyzing customer reviews.
            Generate filter parameters in JSON format with the following structure:
            {
                "rating_range": {"min": float, "max": float},
                "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
                "themes": ["theme1", "theme2"],
                "locations": ["location1", "location2"]
            }
            """
        )
        
        state["filters"] = json.loads(response)
        return state
    except Exception as e:
        raise RuntimeError(f"Filter generation failed: {str(e)}")

async def get_luminoso_stats(state: State) -> State:
    """Get statistics from Luminoso API based on filters."""
    try:
        luminoso_stats = LuminosoStats()
        client = luminoso_stats.initialize_client()
        
        # Get both drivers and sentiment analysis
        drivers = luminoso_stats.fetch_drivers(client)
        sentiment = luminoso_stats.fetch_sentiment(client)
        
        state["luminoso_results"] = {
            "drivers": drivers,
            "sentiment": sentiment
        }
        return state
    except Exception as e:
        raise RuntimeError(f"Luminoso stats retrieval failed: {str(e)}")

async def get_vector_results(state: State, vector_store: Any) -> State:
    """Get relevant reviews from vector store based on filters."""
    try:
        results = vector_store.search(
            state["query"],
            top_k=10,
            filters={
                "rating": state["filters"]["rating_range"],
                "date": state["filters"]["date_range"]
            }
        )
        state["vector_results"] = results
        return state
    except Exception as e:
        raise RuntimeError(f"Vector store search failed: {str(e)}")

async def generate_final_response(state: State, llm_handler: LLMHandler) -> State:
    """Generate final response combining all results."""
    try:
        context = f"""
        Luminoso Analysis Results:
        {json.dumps(state['luminoso_results'], indent=2)}
        
        Vector Store Results:
        {json.dumps(state['vector_results'], indent=2)}
        
        User Query: {state['query']}
        """
        
        response, _ = llm_handler.generate_response(
            state["query"],
            [],  # Context is provided in the prompt
            "claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system_prompt=f"""You are a helpful assistant that provides insights from customer reviews.
            Analyze the following data and provide a comprehensive response:
            
            {context}
            
            Provide a well-structured response that:
            1. Summarizes the key findings
            2. Highlights important trends from Luminoso
            3. Includes specific examples from the vector search results
            4. Provides actionable insights
            """
        )
        
        state["final_response"] = response
        return state
    except Exception as e:
        raise RuntimeError(f"Final response generation failed: {str(e)}")

def create_workflow(llm_handler: LLMHandler, vector_store: Any):
    """Create and return the workflow graph."""
    # Create workflow
    workflow = lg.Graph()
    
    # Add nodes
    workflow.add_node("generate_filters", generate_filters)
    workflow.add_node("get_luminoso_stats", get_luminoso_stats)
    workflow.add_node("get_vector_results", get_vector_results)
    workflow.add_node("generate_final_response", generate_final_response)
    
    # Define edges
    workflow.add_edge("generate_filters", "get_luminoso_stats")
    workflow.add_edge("generate_filters", "get_vector_results")
    workflow.add_edge(("get_luminoso_stats", "get_vector_results"), "generate_final_response")
    
    # Compile workflow
    app = workflow.compile()
    
    return app

def process_query(query: str, llm_handler: LLMHandler, vector_store: Any) -> str:
    """Process a query through the workflow and return the final response."""
    # Initialize workflow
    workflow = create_workflow(llm_handler, vector_store)
    
    # Create initial state
    initial_state = State(
        query=query,
        filters={},
        luminoso_results={},
        vector_results={},
        final_response=""
    )
    
    # Run workflow
    final_state = workflow.invoke(initial_state)
    
    return final_state["final_response"]
