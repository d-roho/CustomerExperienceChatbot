import asyncio
from typing import TypedDict, Dict, Any, List
from anthropic import Anthropic
import json
from utils.llm import LLMHandler
from utils.vector_store import VectorStore
from utils.tools import LuminosoStats
import time
import utils.rag_workflow_funcs


class State(TypedDict):
    query: str
    filters: Dict[str, Any]
    luminoso_results: Dict[str, Any]
    driver_summary: str
    sentiment_summary: str
    vector_results: Dict[str, Any]
    final_response: str
    max_tokens: int
    model: str
    execution_times: Dict[str, float]
    reviews_summary: int
    original_filters: Dict[str, Any]


async def generate_filters(state: State, llm_handler: LLMHandler) -> State:
    """Generate filters based on user query using Claude."""
    try:
        response = llm_handler.anthropic.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=2000,
            temperature=0,
            system=utils.rag_workflow_funcs.generate_filters_prompt,
            messages=[{
                "role": "user",
                "content": f"Query: {state['query']}"
            }])
        print(response.content[0].text)
        state["filters"] = json.loads(response.content[0].text)
        state['original_filters'] = json.loads(response.content[0].text)
        return state
    except Exception as e:
        raise RuntimeError(f"Filter generation failed: {str(e)}")


async def get_luminoso_stats(state: State, llm_handler: Anthropic, summaries: bool = False, themes: bool = True, subsets: bool = True,                             ) -> State:
    """Get statistics from Luminoso API based on filters."""
    try:
        luminoso_start = time.time()
        luminoso_stats = LuminosoStats()
        client = luminoso_stats.initialize_client()

        filter_to_use = state["filters"]
        if themes == 0:
            # remove themes for quicker processing
            filter_to_use = state['filters'].copy()
            filter_to_use['themes'] = []
        if subsets == 0:
            # remove themes for quicker processing
            filter_to_use['subsets'] = []


        # Run API calls concurrently
        drivers_task = asyncio.create_task(luminoso_stats.fetch_drivers(client, filter_to_use))
        sentiment_task = asyncio.create_task(luminoso_stats.fetch_sentiment(client, filter_to_use))
        
        # Wait for both tasks to complete
        drivers_result, sentiment_result = await asyncio.gather(drivers_task, sentiment_task)

        # Extract data dictionaries from results
        drivers_data, dri_exc = drivers_result  # Tuple of (data_dict, execution_time)
        sentiment_data, sent_exc = sentiment_result

        state["luminoso_results"] = {
            "drivers": drivers_data,
            "sentiment": sentiment_data
        }
        state["execution_times"]['driver_stats'] = dri_exc
        state["execution_times"]['sentiment_stats'] = sent_exc


        stats_summary_start = time.time()        
        if summaries == 1:
            
            response = llm_handler.anthropic.messages.create(
                model='claude-3-5-sonnet-20241022',
                max_tokens=2000,
                temperature=0,
                system=utils.rag_workflow_funcs.drivers_summary_prompt,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"User Query: {state['query']} \n Drivers Dataset {json.dumps(state['luminoso_results']['drivers'], indent=2)}"
                }])

            state["driver_summary"] = response.content[0].text

            response = llm_handler.anthropic.messages.create(
                model='claude-3-5-sonnet-20241022',
                max_tokens=2000,
                temperature=0,
                system=utils.rag_workflow_funcs.sentiment_summary_prompt,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"User Query: {state['query']} \n Sentiment Dataset {json.dumps(state['luminoso_results']['sentiment'], indent=2)}"
                }])

            print(response.content[0].text)
            state["sentiment_summary"] = response.content[0].text
        else:

            organized_data = ['The following is data organized by subsets to be used in your analysis. Each subset has data on Drivers and Sentiment for relating only to that subset.']
            driver_keys = state['luminoso_results']['drivers'].keys()
            sentiment_keys = state['luminoso_results']['sentiment'].keys() 

            data_key_tuples = list(zip(driver_keys, sentiment_keys))
            for key in data_key_tuples:
                organized_data.append(f"Subset Metadata: \n Theme: {state['luminoso_results']['drivers'][key[0]]['theme']} \n {state['luminoso_results']['drivers'][key[0]]['subset']}\n\n DRIVERS DATA \n\n {json.dumps(state['luminoso_results']['drivers'][key[0]], indent=2)} \n\n SENTIMENT DATA \n\n {json.dumps(state['luminoso_results']['sentiment'][key[1]], indent=2)} \n\n END OF SUBSET")

            stats_context = '\n\n'.join(organized_data)
            response = llm_handler.anthropic.messages.create(
                model='claude-3-5-sonnet-20241022',
                max_tokens=2000,
                temperature=0,
                system=utils.rag_workflow_funcs.combined_summary_prompt,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"User Query: {state['query']} \n\n {stats_context}"
                }])

            state["driver_summary"] = response.content[0].text

        
        state["execution_times"]["luminoso_stats_summary"]  = time.time() - stats_summary_start
        state["execution_times"]["luminoso_stats_total"] = time.time(
        ) - luminoso_start
        return state
    except Exception as e:
        raise RuntimeError(f"Luminoso stats retrieval failed: {str(e)}")


async def get_vector_results(state: State,
                             vector_store: VectorStore,
                             llm_handler: LLMHandler,
                             top_k: int,
                             reranking: int = 0, subdivide_k: bool = True, reviews_summary: bool = True, subsets: bool = True) -> State:
    """Get relevant reviews from vector store based on filters."""
    try:
        vector_start = time.time()
        filter_to_use = state["filters"]
        if subsets == 0:
            # remove themes for quicker processing
            filter_to_use['subsets'] = []
        results = asyncio.run(vector_store.filter_search(
            filter_to_use,
            state['query'],
            llm_handler,
            top_k=top_k,
            subdivide_k=subdivide_k,
            index_name='reviews-csv-main'))

        state["vector_results"] = results
        print(results)
        if reranking == 1:
            reranked_results = vector_store.rerank_results(
                state['query'], results)
            state["vector_results"] = reranked_results
            print(reranked_results)
        state["execution_times"]["vector_search"] = time.time() - vector_start

        if reviews_summary:
            rev_sum_start = time.time()

            context_text = utils.rag_workflow_funcs.context_generator(
                state['vector_results'])
            response = llm_handler.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                system=utils.rag_workflow_funcs.reviews_summary_prompt,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"""
                    User Query: {state['query']}

                    Relevant Reviews:
                    {context_text}
                    """
                }])

            state["reviews_summary"] = response.content[0].text
            reviews_summary_time = time.time() - rev_sum_start
            state["execution_times"]["reviews_summary_generation"] = reviews_summary_time
            state["execution_times"]["reviews_total"] = state["execution_times"]["vector_search"] +state["execution_times"]["reviews_summary_generation"] 
            
        return state
    except Exception as e:
        raise RuntimeError(f"Vector store search failed: {str(e)}")


async def generate_final_response(state: State,
                                  llm_handler: LLMHandler,
                                  summaries: int = 0, reviews_summary: int = 0) -> State:
    """Generate final response combining all results."""
    try:         
        final_response_start = time.time()

        if reviews_summary:
            context_text = state['reviews_summary']
            
        else:
            context_text = context_generator(state['vector_results'])
        
        stats_summary = f"""
        Drivers Data Analysis:
        {state['driver_summary']}

        Sentiment Data Analysis
        {state['sentiment_summary']}"""

        if summaries == 0:
            stats_summary = f"""
            Data Summary
            {state['driver_summary']}
            """

        response = llm_handler.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=state.get('max_tokens',
                                 2000),  # Updated to use max_tokens from state
            temperature=0,
            system=utils.rag_workflow_funcs.final_answer_prompt,
            messages=[{
                "role":
                "user",
                "content":
                f"""
                User Query: {state['query']}

                {stats_summary}

                Reviews:
                {context_text}
                """
            }])

        state["final_response"] = response.content[0].text
        state["execution_times"]["final_response_generation"] = time.time(
        ) - final_response_start

        return state
    except Exception as e:
        raise RuntimeError(f"Final response generation failed: {str(e)}")


async def process_query(query: str,
                        llm_handler: LLMHandler,
                        vector_store: VectorStore,
                        top_k: int = 200,
                        max_tokens: int = 2000,
                        model: str = "claude-3-5-sonnet-20241022",
                        reranking: int = 1,
                        summaries: int = 0, reviews_summary: int = 0, subdivide_k:bool = False) -> Dict:
    """Process a query through the workflow and return the final response."""
    try:
        import time
        workflow_start = time.time()

        # Initialize state
        state = State(query=query,
                      filters={},
                      luminoso_results={},
                      vector_results={},
                      final_response="",
                      driver_summary="",
                      sentiment_summary="",
                      max_tokens=max_tokens,
                      model=model,
                      reviews_summary="",
                      execution_times={},
                     original_filters={})

        # Step 1: Generate filters
        filter_start = time.time()
        state = await generate_filters(state, llm_handler)
        state["execution_times"]["filter_generation"] = time.time(
        ) - filter_start

        # Steps 2 & 3: Parallel execution of Luminoso stats and vector search
        luminoso_task = asyncio.create_task(
            get_luminoso_stats(state, llm_handler, summaries=summaries))
        vector_task = asyncio.create_task(
            get_vector_results(state,
                               vector_store,
                               llm_handler,
                               top_k=top_k,
                               reranking=reranking, subdivide_k=subdivide_k, 
                               reviews_summary=reviews_summary))

        # Start timing parallel tasks
        parallel_start = time.time()
        results = await asyncio.gather(luminoso_task, vector_task)

        # Record individual task times from results
        state["execution_times"].update(results[0]["execution_times"])
        state["execution_times"].update(results[1]["execution_times"])

        # Record total parallel execution time
        state["execution_times"]["parallel_total"] = time.time() - parallel_start

        # Merge results back into state
        state["luminoso_results"] = results[0]["luminoso_results"]
        state["driver_summary"] = results[0]["driver_summary"]
        if results[0]["sentiment_summary"] != "":
            state["sentiment_summary"] = results[0]["sentiment_summary"]

        state["vector_results"] = results[1]["vector_results"]
        if results[1]["reviews_summary"] != "":
            state["reviews_summary"] = results[1]["reviews_summary"]

        # Step 4: Generate final response
        final_response_start = time.time()
        state = await generate_final_response(state, llm_handler, summaries=summaries, reviews_summary=reviews_summary)
        state["execution_times"]["final_response_generation"] = time.time(
        ) - final_response_start

        # Record total workflow time
        state["execution_times"]["total_workflow"] = time.time(
        ) - workflow_start

        return state
    except Exception as e:
        raise RuntimeError(f"Query processing failed: {str(e)}")


async def process_query_lite(query: str,
    llm_handler: LLMHandler,
    vector_store: VectorStore,
    top_k: int = 200,
    max_tokens: int = 2000,
    model: str = "claude-3-5-sonnet-20241022",
    reranking: int = 0,
    summaries: int = 0, reviews_summary: int = 0, subdivide_k:bool = False) -> Dict:
    """Process a query through the workflow and return the final response."""
    try:
        import time
        workflow_start = time.time()

        # Initialize state
        state = State(query=query,
                      filters={},
                      luminoso_results={},
                      vector_results={},
                      final_response="",
                      driver_summary="",
                      sentiment_summary="",
                      max_tokens=max_tokens,
                      model=model,
                      reviews_summary="",
                      execution_times={},
                     original_filters={})

        # Step 1: Generate filters
        filter_start = time.time()
        state = await generate_filters(state, llm_handler)
        state["execution_times"]["filter_generation"] = time.time(
        ) - filter_start

        # Steps 2 & 3: Parallel execution of Luminoso stats and vector search
        luminoso_task = asyncio.create_task(
            get_luminoso_stats(state,
                               llm_handler, themes=False,
                               subsets=False,
                               summaries=summaries))
        vector_task = asyncio.create_task(
        get_vector_results(state,
               vector_store,
               llm_handler,
               top_k=top_k,
               reranking=reranking, subdivide_k=subdivide_k, 
               reviews_summary=reviews_summary, subsets= False))


        # Start timing parallel tasks
        parallel_start = time.time()
        results = await asyncio.gather(luminoso_task, vector_task)

        # Record individual task times from results
        state["execution_times"].update(results[0]["execution_times"])
        state["execution_times"].update(results[1]["execution_times"])

        # Record total parallel execution time
        state["execution_times"]["parallel_total"] = time.time() - parallel_start

        # Merge results back into state
        state["luminoso_results"] = results[0]["luminoso_results"]
        state["vector_results"] = results[1]["vector_results"]

        # Step 4: Generate final response
        final_response_start = time.time()
        state = await generate_final_response(state, llm_handler, summaries=summaries, reviews_summary=reviews_summary)
        state["execution_times"]["final_response_generation"] = time.time(
        ) - final_response_start

        # Record total workflow time
        state["execution_times"]["total_workflow"] = time.time(
        ) - workflow_start

        return state

    except Exception as e:
        raise RuntimeError(f"Query processing failed: {str(e)}")