from requests.utils import proxy_bypass_registry
import streamlit as st
import os
import time # Added import for time module
from typing import Optional, Tuple, List, Dict, Any
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.llm import LLMHandler
import pandas as pd
from pinecone import ServerlessSpec
import json
from utils.rag_workflow import process_query
from utils.theme_workflow import process_themes
import asyncio
from datetime import datetime

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


# Initialize components with error handling
try:
    llm_handler, vector_store = init_components()
except Exception as e:
    st.error(f"Application initialization failed: {str(e)}")
    st.stop()

# Sidebar controls
st.sidebar.title("Parameters")
# chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500, 50)
# chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, 10)
top_k = st.sidebar.slider("Number of Reviews", 1, 300, 5)
max_tokens = st.sidebar.slider("Max Response Length (tokens)", 100, 4000, 2000)
use_reranking = st.sidebar.checkbox("Use Reranking", True)

# Main interface
st.title("Review Analysis Pipeline")

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
                            'date': chunk['metadata'].get('date', '')
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
                start_time = time.time() #Start timer
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
                    df = pd.DataFrame.from_dict(response['sample_df'])
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
                    execution_time = time.time() - start_time #Stop timer
                    st.subheader(f"Theme Generation Results (Execution time: {execution_time:.2f}s)")
                    with st.expander("Explore Workflow"):
                        st.subheader(f"## Preliminary Themes  w/ Sentiment")
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
selected_tool = st.selectbox("Select Tool",
      ["Luminoso Stats", "Filter Search"],
      index=0)
if selected_tool == "Luminoso Stats":
    pass

elif selected_tool == "Filter Search":
    pass


# Query interface
st.header("Query the Reviews")
query = st.text_input("Enter your query")

if query:
    if st.button("Basic RAG Search"):
        st.session_state.last_query = query
        with st.spinner("Searching..."):
            start_time = time.time() #Start timer
            try:
                # Search for relevant chunks
                print(selected_index)
                results = vector_store.search(query,
                                              llm_handler,
                                              top_k=top_k,
                                              index_name=selected_index)

                # Rerank if enabled
                if use_reranking and results:
                    results = vector_store.rerank_results(query, results)

                # Generate response
                if results:
                    response, context = llm_handler.generate_response(
                        query, results, model, max_tokens=max_tokens)

                    # Display results
                    execution_time = time.time() - start_time #Stop timer
                    st.subheader(f"Generated Response (Execution time: {execution_time:.2f}s)")
                    st.code(response, language="text")

                    st.subheader("Context Used")
                    st.code(context, language="text")
                else:
                    st.warning("No relevant results found")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    if st.button("Fetch Luminoso Stats"):
        from utils.tools import LuminosoStats

        lumin_class = LuminosoStats()
        lumin_client = lumin_class.initialize_client()

        st.session_state.last_query = query
        with st.spinner("Searching..."):
            start_time = time.time() #Start timer
            try:
                with open('attached_assets/test_filter.json', 'r') as file:
                    filter = json.load(file)
                print(filter)

                # fetch drivers
                driver_time = time.time() #Start timer
                drivers = lumin_class.fetch_drivers(lumin_client, filter)
                driver_execution_time = time.time() - driver_time #Stop timer

                sentiment_time = time.time() #Start timer
                sentiment = lumin_class.fetch_sentiment(lumin_client, filter)
                sentiment_execution_time = time.time() - sentiment_time #Stop timer

                # # Display results

                execution_time = time.time() - start_time #Stop timer
                st.success(f'Done (Execution time: {execution_time:.2f}s)')

                st.subheader("Test Filter")
                st.json(filter)

                st.subheader(f"Drivers | Execution Time: {driver_execution_time:.2f}s"")")
                st.dataframe(drivers)

                st.subheader(f"Sentiment | Execution Time: {sentiment_execution_time:.2f}s"")")
                st.dataframe(sentiment)


            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    if st.button("Fetch Filtered Reviews"):
        st.session_state.last_query = query
        with st.spinner("Searching..."):
            start_time = time.time() #Start timer
            try:
                with open('attached_assets/test_filter.json', 'r') as file:
                    filter = json.load(file)
                print(filter)
                # Search for relevant chunks
                print(selected_index)
                results = vector_store.filter_search(filter,
                                                     query,
                                                     llm_handler,
                                                     top_k=top_k,
                                                     index_name=selected_index)

                # Rerank if enabled
                if use_reranking and results:
                    results = vector_store.rerank_results(query, results)

                # Generate response
                if results:
                    response, context = llm_handler.generate_response(
                        query, results, model, max_tokens=max_tokens)

                    # Display results
                    execution_time = time.time() - start_time #Stop timer
                    st.subheader(f"Generated Response (Execution time: {execution_time:.2f}s)")
                    st.code(response, language="text")

                    st.subheader("Context Used")
                    st.code(context, language="text")
                else:
                    st.warning("No relevant results found")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

    if st.button("Agentic Search"):
        st.session_state.last_query = query
        with st.spinner("Processing analysis workflow..."):
            start_time = time.time() #Start timer
            try:
                import asyncio
                response = asyncio.run(
                    process_query(query, llm_handler, vector_store))

                st.subheader("Analysis Results")
                st.markdown(response['final_response'])
                with st.expander("Explore Workflow"):
                    st.subheader(f"## Query: \n {response['query']}")
                    st.subheader(f"## Generated Filter: \n")
                    st.json(response['filters'], expanded=True)
                    st.subheader(f"Drivers Data \n")
                    st.dataframe(response['luminoso_results']['drivers'])
                    st.subheader(f"Drivers Summary \n")
                    st.markdown(response['driver_summary'])
                    st.subheader(f"Sentiment Data \n")
                    st.dataframe(response['luminoso_results']['sentiment'])
                    st.subheader(f"Sentiment Summary \n")
                    st.markdown(response['sentiment_summary'])

                with st.expander(
                        f"{len(response['vector_results'])} Reviews Retrieved"
                ):
                    st.subheader(f"## Total Reviews : \n {response['query']}")
                    for i, curr_rev in enumerate(
                            response['vector_results'][:50]):
                        st.markdown(
                            f"### Review {i+1} | Retriever/Reranker Score: {curr_rev['score']} \n Metadata: {curr_rev['header']} \n\n {curr_rev['text']}"
                        )
                execution_time = time.time() - start_time #Stop timer
                st.success(f'Done (Execution time: {execution_time:.2f}s)')

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Pinecone, and Anthropic Claude")