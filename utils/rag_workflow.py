import asyncio
from typing import TypedDict, Dict, Any, List
from anthropic import Anthropic
import json
from utils.llm import LLMHandler
from utils.vector_store import VectorStore
from utils.tools import LuminosoStats
import time


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


async def generate_filters(state: State, llm_handler: LLMHandler) -> State:
    """Generate filters based on user query using Claude."""
    try:
        response = llm_handler.anthropic.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=2000,
            temperature=0,
            system=
            """Extract metadata from the query in the following format. Return only valid JSON

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

        Current Date in February 2025 
        """,
            messages=[{
                "role": "user",
                "content": f"Query: {state['query']}"
            }])
        print(response.content[0].text)
        state["filters"] = json.loads(response.content[0].text)
        return state
    except Exception as e:
        raise RuntimeError(f"Filter generation failed: {str(e)}")


async def get_luminoso_stats(state: State, llm_handler: Anthropic, themes: 1,
                             summaries: 0) -> State:
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

        # Run API calls concurrently
        drivers_task = asyncio.create_task(luminoso_stats.fetch_drivers(client, filter_to_use))
        sentiment_task = asyncio.create_task(luminoso_stats.fetch_sentiment(client, filter_to_use))
        
        # Wait for both tasks to complete
        drivers, sentiment = await asyncio.gather(drivers_task, sentiment_task)

        state["luminoso_results"] = {
            "drivers": drivers.to_dict(),
            "sentiment": sentiment.to_dict()
        }

        if summaries == 0:
            response = llm_handler.anthropic.messages.create(
                model='claude-3-5-sonnet-20241022',
                max_tokens=2000,
                temperature=0,
                system="""
                You are an expert customer experience analyst that gathers insights from aggregate rating drivers data. Your task is to analyze the following data and make a thorough summary to be used to understand the overall customer experience. Align your summary with the User Query so as to cover everything that may be required for answering the query. Assume the data provided is for the date and locations etc. asked by the user

                In your analysis, Be sure to include statistics to back up your analysis. Here is an explanation of the data:

                Dataset - A collection of themes (aspects of customer experience) and their impact on customer ratings. 

                Theme Name: Name of the aspect
                Normalized Relevance to Subset: The proportion of the total number of customers that this theme is relevant to (0-1). This is an indicator of pervasiveness of the theme.
                Matches Found in Subset: Number of reviews that include this theme
                Impact on Score: effect of this theme on the customer ratings (-1 to +1)
                Impact Confidence Level: The confidence level of the impact on score. This is an indicator of the reliability of the impact on score. (0-1)
                Average Score: The average rating of the customer reviews that include this theme. (0-1)
                Baseline Score: The average rating of all customers.(0-1)            

            """,
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
                system="""
                You are an expert customer experience analyst that gathers insights from aggregate sentiment data. Your task is to analyze the following data and make a thorough summary to be used to understand the overall customer experience. Align your summary with the User Query so as to cover everything that may be required for answering the query. Assume the data provided is for the date and locations etc. asked by the user

                In your analysis, Be sure to include statistics to back up your analysis. Here is an explanation of the data:

                Dataset - A collection of themes (aspects of customer experience) and customer sentiment around those themes. 

                Theme Name: Name of the aspect
                Proportion of Subset With Theme: The proportion of the total reviews that include this theme (0-1). This is an indicator of pervasiveness of the theme.)
                Proportion of Positive Mentions: The proportion theme mentions made with a positive sentiment (0-1)
                Proportion of Neutral Mentions: The proportion theme mentions made with a neutral sentiment (0-1)
                Proportion of NEgative Mentions: The proportion theme mentions made with a neutral sentiment (0-1)s
            """,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"User Query: {state['query']} \n Sentiment Dataset {json.dumps(state['luminoso_results']['sentiment'], indent=2)}"
                }])

            print(response.content[0].text)
            state["sentiment_summary"] = response.content[0].text
        else:
            response = llm_handler.anthropic.messages.create(
                model='claude-3-5-sonnet-20241022',
                max_tokens=2000,
                temperature=0,
                system="""
                You are an expert customer experience analyst that gathers insights from aggregate data. Your task is to analyze the following data and make a thorough summary to be used to understand the overall customer experience. Align your summary with the User Query so as to cover everything that may be required for answering the query. Assume the data provided is for the date and locations etc. asked by the user

                In your analysis, Be sure to include statistics to back up your analysis. Here is an explanation of the data:

Drivers Dataset - A collection of themes (aspects of customer experience) and their impact on customer ratings. 

Theme Name: Name of the aspect
Normalized Relevance to Subset: The proportion of the total number of customers that this theme is relevant to (0-1). This is an indicator of pervasiveness of the theme.
Matches Found in Subset: Number of reviews that include this theme
Impact on Score: effect of this theme on the customer ratings (-1 to +1)
Impact Confidence Level: The confidence level of the impact on score. This is an indicator of the reliability of the impact on score. (0-1)
Average Score: The average rating of the customer reviews that include this theme. (0-1)
Baseline Score: The average rating of all customers.(0-1)            

                Sentiment Dataset - A collection of themes (aspects of customer experience) and customer sentiment around those themes. 

                Theme Name: Name of the aspect
                Proportion of Subset With Theme: The proportion of the total reviews that include this theme (0-1). This is an indicator of pervasiveness of the theme.)
                Proportion of Positive Mentions: The proportion theme mentions made with a positive sentiment (0-1)
                Proportion of Neutral Mentions: The proportion theme mentions made with a neutral sentiment (0-1)
                Proportion of Negative Mentions: The proportion theme mentions made with a neutral sentiment (0-1)s
            """,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"User Query: {state['query']} \n Drivers Dataset\n {json.dumps(state['luminoso_results']['drivers'], indent=2)} \n Sentiment Dataset\n {json.dumps(state['luminoso_results']['sentiment'], indent=2)}"
                }])

            state["driver_summary"] = response.content[0].text

        state["execution_times"]["luminoso_stats"] = time.time(
        ) - luminoso_start
        return state
    except Exception as e:
        raise RuntimeError(f"Luminoso stats retrieval failed: {str(e)}")


async def get_vector_results(state: State,
                             vector_store: VectorStore,
                             llm_handler: LLMHandler,
                             top_k: int,
                             reranking: int = 1) -> State:
    """Get relevant reviews from vector store based on filters."""
    try:
        vector_start = time.time()
        results = vector_store.filter_search(
            state["filters"],
            state['query'],
            llm_handler,
            top_k=top_k,  # Default value, can be made configurable
            index_name='reviews-csv-main')

        state["vector_results"] = results
        print(results)
        if reranking == 1:
            reranked_results = vector_store.rerank_results(
                state['query'], results)
            state["vector_results"] = reranked_results
            print(reranked_results)
        state["execution_times"]["vector_search"] = time.time() - vector_start
        return state
    except Exception as e:
        raise RuntimeError(f"Vector store search failed: {str(e)}")


async def generate_final_response(state: State,
                                  llm_handler: LLMHandler,
                                  summaries: int = 0, reviews_summary: int = 0) -> State:
    """Generate final response combining all results."""
    try:         
        final_response_start = time.time()

        context_text = "\n".join([
            f" Review {idx} (Retriever Score: {c['score']}) \nMetadata: {c['header']} \n - Text: {c['text']}\n\n"
            for idx, c in enumerate(state['vector_results'])
        ])


        if reviews_summary == 1:
            response = llm_handler.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                temperature=0,
                system=
                """You are a helpful customer experience analysis expert that provides insights from raw customer reviews. Provide a comprehensive analysis of the customer experience based on the USER QUERY and the raw reviews. Be sure to direct quotes from reviews to back up your analysis everywhere possible.   

                Bullet points should be used wherever necessary
                Reference the source text wherever using them to answer the question
                """,
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
            reviews_summary_time = time.time() - final_response_start
            state["execution_times"]["reviews_summary_generation"] = reviews_summary_time
            context_text = response.content[0].text
            print(context_text)

        stats_summary = f"""
        Drivers Data Analysis:
        {state['driver_summary']}

        Sentiment Data Analysis
        {state['sentiment_summary']}"""

        if summaries == 1:
            stats_summary = f"""
            Data Summary
            {state['driver_summary']}
            """


        response = llm_handler.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=state.get('max_tokens',
                                 2000),  # Updated to use max_tokens from state
            temperature=0,
            system=
            """You are a helpful customer experience analysis expert that provides insights from aggregate data summaries and customer reviews. Combine the data summaries and customer reviews to provide a comprehensive analysis of the customer experience. Be sure to include statistics and direct quotes from reviews to back up your analysis everywhere possible.   

            Provide a well-structured response that includes:
            A basic line to summarize the question and give a start to the answer. Keep it neutral and informative

            Bullet points should be used wherever necessary
            Reference the source wherever using them to answer the question
            Use numbers and percentages when necessary

            Summary of the answer in 2-3 lines emphasizing the most important points.
            Suggested actions based on findings and query if necessary
            Provide potential follow up query to answer if necessary
            """,
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
                        top_k: int = 300,
                        max_tokens: int = 2000,
                        model: str = "claude-3-5-sonnet-20241022",
                        reranking: int = 1,
                        summaries: int = 0, reviws_summary: int = 0) -> Dict:
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
                      reviews_summary=reviws_summary,
                      execution_times={})

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
                               reranking=reranking))

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


async def process_query_lite(query: str,
                             llm_handler: LLMHandler,
                             vector_store: VectorStore,
                             top_k: int = 300,
                             max_tokens: int = 2000,
                             model: str = "claude-3-5-sonnet-20241022",
                             themes: int = 0,
                             summaries: int = 0,
                             reranking: int = 1,
                             reviews_summary: int = 0) -> Dict:
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
                      reviews_summary=reviews_summary,
                      execution_times={})

        # Step 1: Generate filters
        filter_start = time.time()
        state = await generate_filters(state, llm_handler)
        state["execution_times"]["filter_generation"] = time.time(
        ) - filter_start

        # Steps 2 & 3: Parallel execution of Luminoso stats and vector search
        luminoso_task = asyncio.create_task(
            get_luminoso_stats(state,
                               llm_handler,
                               themes=themes,
                               summaries=summaries))
        vector_task = asyncio.create_task(
            get_vector_results(state,
                               vector_store,
                               llm_handler,
                               top_k=top_k,
                               reranking=reranking))

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