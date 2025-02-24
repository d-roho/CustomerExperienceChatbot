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
import utils.app_funcs 

# Initialize session state
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

# Page configuration
st.set_page_config(page_title="RAG Pipeline for Reviews", layout="wide")

# Model selection in sidebar
st.sidebar.title("Model Settings")
model = st.sidebar.selectbox(
    "Select Model",
    ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
    index=1  # Default to Sonnet
)


# Initialize components
@st.cache_resource
def initialize_components():
    return utils.app_funcs.init_components()

# Initialize components with error handling
try:
    llm_handler, vector_store = initialize_components()
except Exception as e:
    st.error(f"Application initialization failed: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.title("Parameters")
# chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500, 50)
# chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, 10)
top_k = st.sidebar.slider("Number of Reviews", 1, 300, 200)
subdivide_k = st.sidebar.checkbox("Divide K by Number of Subsets", True)
max_tokens = st.sidebar.slider("Max Final Response Length (tokens)", 100, 4000, 2000)
use_reranking = st.sidebar.checkbox("Use Reranking", False)

# Main interface
st.title("Review Analysis Pipeline")

st.header("Vector Store and Theme Generation")
# Input selection method
input_method = st.radio("Select Input Method",
                        ["Existing Vector Store", "File Upload"],
                        index=0)  # Default to Existing Vector Store

if input_method == "File Upload":
    # File upload
    uploaded_file = st.file_uploader("Upload Reviews File",
                                     type=['txt', 'csv'])
    index_name = st.text_input("Enter Pinecone Index Name", "reviews-csv-test")

    process_button = st.button("Process File")

    if uploaded_file and process_button:
        file_type = uploaded_file.name.split('.')[-1].lower()

        # Update vector store index
        try:
            # Delete index if it exists
            if index_name in vector_store.pc.list_indexes().names():
                vector_store.pc.delete_index(index_name)
                st.info(f"Deleted existing index: {index_name}")

            # Create new index
            vector_store.pc.create_index(name=index_name,
                                         dimension=vector_store.dimension,
                                         metric='cosine',
                                         spec=ServerlessSpec(
                                             cloud='aws',
                                             region=vector_store.environment))
            st.success(f"Created new index: {index_name}")
            vector_store.index = vector_store.pc.Index(index_name)
            vector_store.index_name = index_name
            selected_index = index_name
        except Exception as e:
            st.error(f"Failed to create/switch index: {str(e)}")
            st.stop()

        # Process the file
        with st.spinner("Processing file..."):
            try:
                # Initialize text processor
                text_processor = TextProcessor(chunk_size=500,
                                               chunk_overlap=50)

                # Process file based on type
                content = uploaded_file.read()
                chunks, df = text_processor.process_file(content, file_type)
                st.session_state.processed_chunks = chunks

                # Display preview
                if file_type == 'csv':
                    st.subheader("Preview of processed reviews")
                    preview_df = pd.DataFrame([
                        {
                            'text': chunk['text'],
                            'city': chunk['metadata'].get(
                                'location', ''),  #Quebec missing in dataset
                            'rating': chunk['metadata'].get('rating', ''),
                            'date_unix': chunk['metadata'].get('date', '')
                        } for chunk in chunks[:5]
                    ])
                    st.dataframe(preview_df)
                else:
                    st.subheader("Preview of text chunks")
                    for chunk in chunks[:5]:
                        st.text(chunk['text'][:200] + "...")

                # Store in vector database
                try:
                    vector_store.upsert_texts(chunks, llm_handler, index_name,
                                              df)
                    st.success(f"Processed {len(chunks)} chunks from the file")
                except Exception as e:
                    st.error(
                        f"Failed to store chunks in vector database: {str(e)}")

            except Exception as e:
                st.error(f"Failed to process file: {str(e)}")

elif input_method == "Existing Vector Store":
    # Get available indexes
    available_indexes = vector_store.pc.list_indexes().names()

    if available_indexes:
        default_index = available_indexes.index(
            'reviews-csv-main'
        ) if 'reviews-csv-main' in available_indexes else 0
        selected_index = st.selectbox("Select Vector Store",
                                      available_indexes,
                                      index=default_index)
        if st.button("Generate Themes"):
            with st.spinner("Searching..."):
                start_time = time.time()  #Start timer
                try:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    # Generate themes using theme_workflow
                    response = asyncio.run(
                        process_themes(selected_index, llm_handler,
                                       vector_store))
                except Exception as e:
                    st.error(f"Failed to generate themes: {str(e)}")

                try:
                    # save results
                    output_file = f'attached_assets/saved_output/reviews_tagged_{timestamp}.csv'
                    df = pd.DataFrame.from_dict(response['tagged_df'])
                    df.to_csv(output_file, index=False)

                    with open(
                            f"attached_assets/saved_output/preliminary_themes_{timestamp}.json",
                            "w") as outfile:
                        json.dump(response['preliminary_themes'], outfile)

                    with open(
                            f"attached_assets/saved_output/refined_themes_{timestamp}.json",
                            "w") as outfile:
                        json.dump(response['refined_themes'], outfile)

                    print(
                        f"Output saved to attached_assets/saved_output | Timestamp: {timestamp}"
                    )

                    # Display results
                    execution_time = time.time() - start_time  #Stop timer
                    st.subheader(
                        f"Theme Generation Results (Execution time: {execution_time:.2f}s)"
                    )
                    with st.expander("Explore Workflow"):
                        st.subheader(f"reliminary Themes  w/ Sentiment")
                        st.json(response['preliminary_themes'], expanded=True)
                        st.subheader(f"Merged & Refine Themes \n")
                        st.json(response['refined_themes'], expanded=True)
                        st.subheader(
                            f"Reviews tagged with Themes and Subthemes")
                        st.dataframe(df[['Text', 'Theme_Subthemes']])
                except Exception as e:
                    st.error(f"Failed to output results: {str(e)}")

        if selected_index != vector_store.index_name:
            vector_store.index = vector_store.pc.Index(selected_index)
            vector_store.index_name = selected_index
            index_name = vector_store.index_name
            st.success(f"Connected to index: {selected_index}")

    else:
        st.warning("No existing vector stores found")

# Initialize session state for results
if 'results' not in st.session_state:
    st.session_state.results = None
if 'response' not in st.session_state:
    st.session_state.response = None

# Query interface
st.header("Test the Tools")

st.subheader("Filter Parameters")
col1, col2 = st.columns(2)

with col2:
    # Cities
    cities_list = utils.app_funcs.cities_list
    select_all_cities = st.checkbox("Select All Cities",
                                    key="select_all_cities")
    selected_cities = st.multiselect(
        "Cities",
        cities_list,
        default=cities_list if select_all_cities else ["New York", "Los Angeles"],
        key="cities_select")

    # States
    states_list = utils.app_funcs.states_list
    select_all_states = st.checkbox("Select All States",
                                    key="select_all_states")
    selected_states = st.multiselect(
        "States",
        states_list,
        default=states_list if select_all_states else ['NY', 'CA'],
        key="states_select")

    # Locations
    locations_list = utils.app_funcs.locations_list
    select_all_locations = st.checkbox("Select All Locations",
                                       key="select_all_locations")
    selected_location = st.multiselect(
        "Select Store Locations",
        locations_list,
        default=locations_list if select_all_locations else ["43 Spring St, New York, NY", "8404 Melrose Ave, Los Angeles, CA"],
        key="location_select")

    # Themes
    themes_list = utils.app_funcs.themes_list
    select_all_themes = st.checkbox("Select All Themes",
                                    key="select_all_themes")
    selected_themes = st.multiselect(
        "Select Themes",
        themes_list,
        default=themes_list if select_all_themes else ["Exceptional Customer Service & Support","Poor Service & Long Wait Times"],
        key="themes_select")

with col1:
    start_month = st.number_input("Start Month",
                                  min_value=1,
                                  max_value=12,
                                  value=1,
                                  key="start_month")
    start_year = st.number_input("Start Year",
                                 min_value=2019,
                                 max_value=2025,
                                 value=2019,
                                 key="start_year")
    end_month = st.number_input("End Month",
                                min_value=1,
                                max_value=12,
                                value=12,
                                key="end_month")
    end_year = st.number_input("End Year",
                               min_value=2019,
                               max_value=2025,
                               value=2025,
                               key="end_year")

    min_rating = st.number_input("Minimum Rating",
                                 min_value=1,
                                 max_value=5,
                                 value=1,
                                 key="min_rating")
    max_rating = st.number_input("Minimum Rating",
                                 min_value=1,
                                 max_value=5,
                                 value=5,
                                 key="max_rating")

subsets = st.multiselect("Select Attributes to Subset Data on",
               ['year','cities', 'states', 'location', 'themes'],
                         default = ['states', 'themes', 'location', 'cities'],
               key='subsets_select')
selected_tool = st.selectbox(
    "Select Tool",
    ["Luminoso Stats API", "Basic RAG Search", "Metadata Filter RAG Search"],
    index=2)
if selected_tool == "Luminoso Stats API":

    if st.button("Fetch Stats", key="fetch_stats"):
        filter_params = {
            "cities": selected_cities,
            "states": selected_states,
            "themes": selected_themes,
            "location": selected_location,
            "month_start": [start_month],
            "year_start": [start_year],
            "month_end": [end_month],
            "year_end": [end_year],
            'subsets': subsets
        }

        lumin_class = LuminosoStats()
        lumin_client = lumin_class.initialize_client()

        with st.spinner("Searching..."):
            start_time = time.time()  #Start timer
            try:
                print(filter_params)

                async def run_tasks():
                    lumin_client = LuminosoStats().initialize_client()
                    drivers_task = asyncio.create_task(
                        LuminosoStats().fetch_drivers(lumin_client,
                                                      filter_params))
                    sentiment_task = asyncio.create_task(
                        LuminosoStats().fetch_sentiment(
                            lumin_client, filter_params))
                    results = await asyncio.gather(drivers_task,
                                                   sentiment_task)
                    return results[0][0], results[0][1], results[1][
                        0], results[1][1]

                drivers, driver_time, sentiment, sentiment_time = asyncio.run(
                    run_tasks())

                # Calculate execution times
                execution_time = time.time() - start_time

                # Display results

                execution_time = time.time() - start_time  #Stop timer
                st.success(f'Done (Execution time: {execution_time:.2f}s)')

                st.subheader("Filter")
                st.json(filter_params)

                st.subheader(f"Drivers | Execution Time: {driver_time:.2f}s"
                             ")")

                for key, value in drivers.items():
                    st.subheader(f"Theme {key}: {value['theme']} | Subset: {value['subset']}")
                    st.dataframe(value['df'])

                st.subheader(
                    f"Sentiment | Execution Time: {sentiment_time:.2f}s"
                    ")")
                for key, value in sentiment.items():
                    st.subheader(f"Theme {key}: {value['theme']}  | Subset: {value['subset']}")
                    st.dataframe(value['df'])

            except Exception as e:
                st.error(f"Luminoso Stats Retrieval failed: {str(e)}")

elif selected_tool == "Basic RAG Search":
    query_basic = st.text_input("Enter the query")
    if st.button("Retrieve Reviews", key="retrieve_reviews"):

        with st.spinner("Searching..."):
            try:
                start_time = time.time()  #Start timer
                results = vector_store.search(query_basic,
                                              llm_handler,
                                              top_k=top_k,
                                              index_name=selected_index)
                pinecone_execution_time = time.time() - start_time  #Stop timer

                # Rerank if enabled
                rerank_time = time.time()  #Start timer
                rerank_execution_time = 0
                if use_reranking and results:
                    results = asyncio.run(vector_store.rerank_results(query_basic, results))
                    rerank_execution_time = time.time(
                    ) - rerank_time  #Stop timer

                # Generate response
                if results:
                    response, context = llm_handler.generate_response(
                        query_basic, results, model, max_tokens=max_tokens)

                    # Display results
                    execution_time = time.time() - start_time  #Stop timer
                    st.subheader(
                        f"Generated Response (Execution time: {execution_time:.2f}s)"
                    )
                    st.code(response, language="text")

                    st.subheader(
                        f"Context Used | Pinecone Retrieval Time: {pinecone_execution_time:.2f}s | Reranking Time: {rerank_execution_time:.2f}s"
                    )
                    st.code(context, language="text")
                else:
                    st.warning("No relevant results found")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

elif selected_tool == "Metadata Filter RAG Search":
    query_filter = st.text_input("Enter the query")
    if st.button("Retrieve Reviews", key="retrieve_reviews"):
        filter_params = {
            "cities": selected_cities,
            "states": selected_states,
            "themes": selected_themes,
            "location":selected_location,
            "month_start": [start_month],
            "year_start": [start_year],
            "month_end": [end_month],
            "year_end": [end_year],
            'subsets': subsets
        }

        with st.spinner("Searching..."):
            try:
                start_time = time.time()  #Start timer
                results = asyncio.run(vector_store.filter_search(filter_params,
                                                          query_filter,
                                                          llm_handler,
                                                          top_k=top_k, subdivide_k=subdivide_k,
                                                          index_name=selected_index))
                pinecone_execution_time = time.time() - start_time  #Stop timer

                # Rerank if enabled
                rerank_time = time.time()  # Start timer
                rerank_execution_time = 0
                if use_reranking and results:
                    try:

                        async def rerank_data(subset):
                            return await vector_store.rerank_results(query_filter, subset['processed_results'])

                        tasks = [rerank_data(subset_data) for subset_data in results.values()]
                        processed_results = asyncio.run(asyncio.gather(*tasks))

                        for (key, subset_data), processed in zip(results.items(), processed_results):
                            subset_data['processed_results'] = processed

                    except Exception as e:
                        st.error(f"Reranking failed: {str(e)}")

                    rerank_execution_time = time.time() - rerank_time  # Stop timer

                # Generate response
                if results:
                    response, context = llm_handler.generate_response(
                        query_filter, results, model, max_tokens=max_tokens)

                    for i in range(1):
                        print(results[1])

                    # Display results
                    execution_time = time.time() - start_time  #Stop timer
                    st.subheader(
                        f"Generated Response (Execution time: {execution_time:.2f}s)"
                    )
                    st.code(response, language="text")

                    st.subheader(
                        f"Context Used | Pinecone Retrieval Time: {pinecone_execution_time:.2f}s | Reranking Time: {rerank_execution_time:.2f}s"
                    )
                    st.code(context, language="text")
                else:
                    st.warning("No relevant results found")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

# Query interface
st.header("Run Agentic RAG \n (Luminoso Stats + Filtered Reviews + Prompting)")
query = st.text_input("Enter your query")
reviews_summary = st.checkbox("Generate & Use summary for Reviews", True)
concise_mode = st.checkbox("Concise Answer Mode", True) # Added Concise Answer Mode checkbox
sep_summary = st.checkbox("Generate summaries separately for Drivers and Sentiment",
                             False)
subset = st.checkbox("Divide by Subsets)", False)


if st.button("Run Workflow"):
    st.session_state.last_query = query
    with st.spinner("Processing analysis workflow..."):
        start_time = time.time()  #Start timer
        try:
            import asyncio
            if not subset:
                response = asyncio.run(
                    process_query_lite(query,
                                       llm_handler,
                                       vector_store,
                                       top_k=top_k,
                                       max_tokens=max_tokens,
                                       model=model,
                                       reranking=use_reranking,
                                       summaries=sep_summary,
                                       reviews_summary=reviews_summary,
                                       concise_mode=concise_mode, #added concise_mode
                                      subdivide_k=subdivide_k))
                lite_execution = 1
            else:
                response = asyncio.run(
                    process_query(query,
                                  llm_handler,
                                  vector_store,
                                  top_k=top_k,
                                  max_tokens=max_tokens,
                                  model=model,
                                  reranking=use_reranking,
                                  summaries=sep_summary,
                                  reviews_summary=reviews_summary,
                                  concise_mode=concise_mode, #added concise_mode
                                 subdivide_k=subdivide_k))
                lite_execution = 0

            st.subheader("Analysis Results")
            st.markdown(response['final_response'])
            with st.expander("Explore Workflow"):
                st.subheader(
                    f"Execution Times (Lite Version: {lite_execution == 1} | Single Stats Summary: {sep_summary == 0})"
                )
                for step, duration in response['execution_times'].items():
                    st.metric(f"{step.replace('_', ' ').title()}",
                              f"{duration:.2f}s")

                st.subheader(f"Query: \n {response['query']}")
                st.subheader(f"Generated Filter: \n")
                st.json(response['original_filters'], expanded=True)
                st.subheader(f"Drivers Data \n")
                for key, value in response['luminoso_results']['drivers'].items():
                    headers = [f"Themes: {value['theme']}"]
                    for sub in value['subset']:
                        try:
                            if sub == 'Overall Rating':
                                headers.append(f"Rating: {sub['minimum']}-{sub['maximum']} stars")
                            elif sub == 'Date Created':
                                start = datetime.fromtimestamp({sub['minimum']})
                                end = datetime.fromtimestamp({sub['maximum']})
                                headers.append(f"Date Rage: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

                            elif sub.get('minimum'):
                                headers.append(f"{sub['name']}: {sub['minimum']} to {sub['maximum']}")

                            else:
                                headers.append(f"{sub['name']}: {sub.get('values')}")
                            header_final = " | ".join(headers)
                        except:
                            header_final = "Failed to retrieve header" 

                    st.subheader(f"{header_final}")
                    st.dataframe(value['df'])
                if sep_summary == 1:
                    st.subheader(f"Drivers Summary \n")
                    st.markdown(response['driver_summary'])
                st.subheader(f"Sentiment Data \n")
                for key, value in response['luminoso_results']['sentiment'].items():
                    headers = [f"Themes: {value['theme']}"]
                    for sub in value['subset']:
                        try:
                            if sub == 'Overall Rating':
                                headers.append(f"Rating: {sub['minimum']}-{sub['maximum']} stars")
                            elif sub == 'Date Created':
                                start = datetime.fromtimestamp({sub['minimum']})
                                end = datetime.fromtimestamp({sub['maximum']})
                                headers.append(f"Date Rage: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

                            elif sub.get('minimum'):
                                headers.append(f"{sub['name']}: {sub['minimum']} to {sub['maximum']}")

                            else:
                                headers.append(f"{sub['name']}: {sub.get('values')}")
                            header_final = " | ".join(headers)
                        except:
                            header_final = "Failed to retrieve header" 

                    st.subheader(f"{header_final}")
                    st.dataframe(value['df'])
                if sep_summary == 1:
                    st.subheader(f"Sentiment Summary \n")
                    st.markdown(response['sentiment_summary'])
                if sep_summary == 0:
                    st.subheader(f"Combined Luminoso Stats Summary \n")
                    st.markdown(response['driver_summary'])


            # reviews
            try:
                rev_count = [len(revset['processed_results']) for _, revset in response['vector_results'].items()]
            except:
                rev_count = [0, 0]
            with st.expander(
                    f"{sum(rev_count)} Reviews Retrieved across {len(response['vector_results'])} Subsets"):
                st.subheader(f"Query:{response['query']} \n______________________________________________")
                if reviews_summary == 1:
                    st.subheader("Reviews Summary")
                    st.markdown(response['reviews_summary'] + '\n______________________________________________')
                for key, revset in response['vector_results'].items():
                    st.subheader(
                    f"Subset {key+1} Info - {revset['subset_info']} | Total Reviews: {len(revset['processed_results'])}")
                    for i, curr_rev in enumerate(revset['processed_results'][:50]):
                        st.markdown(
                            f"Review {i+1} | Retriever/Reranker Score: {curr_rev['score']} \n\n Metadata: {curr_rev['header']} \n\n {curr_rev['text']} \n______________________________________________"
                            )
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Pinecone, and Anthropic Claude")