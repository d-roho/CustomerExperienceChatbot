import streamlit as st
import os
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
def init_components():
    llm_handler = LLMHandler()
    vector_store = VectorStore(
        api_key=os.environ.get('PINECONE_API_KEY'),
        environment=os.environ.get('PINECONE_ENVIRONMENT'),
        index_name='reviews-index'
    )
    return llm_handler, vector_store

llm_handler, vector_store = init_components()

# Sidebar controls
st.sidebar.title("Parameters")
chunk_size = st.sidebar.slider("Chunk Size", 100, 1000, 500, 50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, 10)
top_k = st.sidebar.slider("Number of Results", 1, 10, 5)
use_reranking = st.sidebar.checkbox("Use Reranking", True)

# Main interface
st.title("Review Analysis Pipeline")

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
        vector_store.upsert_texts(chunks, llm_handler.get_client())
        
        st.success(f"Processed {len(chunks)} chunks from the file")

# Query interface
st.header("Query the Reviews")
query = st.text_input("Enter your query")

if query:
    with st.spinner("Searching..."):
        # Search for relevant chunks
        results = vector_store.search(
            query,
            llm_handler.get_client(),
            top_k=top_k
        )
        
        # Rerank if enabled
        if use_reranking:
            results = vector_store.rerank_results(
                query,
                results,
                llm_handler.get_client()
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

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Pinecone, and Anthropic Claude")
