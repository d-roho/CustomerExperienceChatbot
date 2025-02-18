import streamlit as st
import os
from typing import Optional, Tuple, List, Dict, Any
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.llm import LLMHandler
import pandas as pd
from pinecone import ServerlessSpec
from utils.workflow import process_query

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
    index=0  # Default to Haiku
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
use_reranking = st.sidebar.checkbox("Use Reranking", True)
max_tokens = st.sidebar.slider("Max Response Length (tokens)", 100, 4000, 2000)

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
            vector_store.pc.create_index(
                name=index_name,
                dimension=vector_store.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws',
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
            'reviews-csv-test'
        ) if 'reviews-csv-test' in available_indexes else 0
        selected_index = st.selectbox("Select Vector Store",
                                      available_indexes,
                                      index=default_index)

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
st.header("Query the Reviews")
query = st.text_input("Enter your query")

if query:
    if st.button("Search"):
        st.session_state.last_query = query
        with st.spinner("Searching..."):
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
                    st.subheader("Generated Response")
                    st.code(response, language="text")

                    st.subheader("Context Used")
                    st.code(context, language="text")
                else:
                    st.warning("No relevant results found")

            except Exception as e:
                st.error(f"Search failed: {str(e)}")

# Add to existing code after query interface section
st.header("Advanced Query Analysis")
advanced_query = st.text_input("Enter your query for detailed analysis")

if advanced_query:
    if st.button("Analyze"):
        with st.spinner("Processing analysis workflow..."):
            try:
                response = process_query(advanced_query, llm_handler, vector_store)

                st.subheader("Analysis Results")
                st.markdown(response)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Pinecone, and Anthropic Claude")