import streamlit as st
import os
from typing import Optional, Tuple
from utils.text_processor import TextProcessor
from utils.vector_store import VectorStore
from utils.llm import LLMHandler

# Initialize session state
if 'processed_chunks' not in st.session_state:
    st.session_state.processed_chunks = []

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline for Reviews",
    layout="wide"
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
            st.error("Missing required Pinecone credentials. Please check your environment variables.")
            st.stop()

        try:
            vector_store = VectorStore(
                api_key=api_key,
                environment=environment,
                index_name='reviews-index'
            )
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
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500, 50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, 10)
top_k = st.sidebar.slider("Number of Results", 1, 10, 5)
use_reranking = st.sidebar.checkbox("Use Reranking", True)

# Main interface
st.title("Review Analysis Pipeline")

# Input selection method
input_method = st.radio("Select Input Method", ["File Upload", "Existing Vector Store"])

if input_method == "File Upload":
    # File upload
    uploaded_file = st.file_uploader("Upload Reviews Text File", type=['txt'])
    
    if uploaded_file:
        with st.spinner("Processing file..."):
            # Read and process the file
            text_processor = TextProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            content = uploaded_file.read().decode()
            chunks = text_processor.process_file(content)
            st.session_state.processed_chunks = chunks
    
            # Store in vector database
            try:
                vector_store.upsert_texts(chunks, llm_handler)
            st.success(f"Processed {len(chunks)} chunks from the file")
            except Exception as e:
        vector_store.upsert_texts(chunks, llm_handler)
    
elif input_method == "Existing Vector Store":
    # Get available indexes
    available_indexes = vector_store.pc.list_indexes().names()
    
    if available_indexes:
        selected_index = st.selectbox("Select Vector Store", available_indexes)
        if selected_index != vector_store.index_name:
            vector_store.index = vector_store.pc.Index(selected_index)
            vector_store.index_name = selected_index
            st.success(f"Connected to index: {selected_index}")
    else:
        st.warning("No existing vector stores found")

# Query interface
st.header("Query the Reviews")
query = st.text_input("Enter your query")

if query:
    with st.spinner("Searching..."):
        try:
            # Search for relevant chunks
            results = vector_store.search(
                query,
                llm_handler,
                top_k=top_k
            )

            # Rerank if enabled
            if use_reranking:
                results = vector_store.rerank_results(
                    query,
                    results,
                    llm_handler
                )

            # Generate response
            response = llm_handler.generate_response(query, results)

            # Display results
            st.subheader("Generated Response")
            st.write(response)

            st.subheader("Relevant Passages")
            for i, result in enumerate(results, 1):
                with st.expander(f"Passage {i} (Score: {result['score']:.4f})"):
                    st.write(result['text'])

        except Exception as e:
            st.error(f"Search failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Pinecone, and Anthropic Claude")