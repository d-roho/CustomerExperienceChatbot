import streamlit as st
from components.api_operations import render_api_operations
from components.visualizations import render_visualizations
from utils.config import load_config, save_config
from utils.api_client import initialize_client

def main():
    st.set_page_config(
        page_title="Luminoso API Interface",
        page_icon="🔆",
        layout="wide"
    )

    st.title("Luminoso API Interface")

    # Configuration section
    with st.sidebar:
        st.header("API Configuration")
        config = load_config()
        
        api_url = st.text_input("API URL", value=config.get("api_url", ""), type="default")
        api_token = st.text_input("API Token", value=config.get("api_token", ""), type="password")
        
        if st.button("Save Configuration"):
            save_config({
                "api_url": api_url,
                "api_token": api_token
            })
            st.success("Configuration saved!")

    # Initialize API client
    try:
        client = initialize_client()
        
        # Main content tabs
        tab1, tab2 = st.tabs(["API Operations", "Visualizations"])
        
        with tab1:
            render_api_operations(client)
        
        with tab2:
            render_visualizations(client)

    except Exception as e:
        st.error(f"Failed to initialize API client: {str(e)}")
        st.info("Please check your API configuration in the sidebar.")

if __name__ == "__main__":
    main()
