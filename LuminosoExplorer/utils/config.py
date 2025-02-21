import streamlit as st
import json
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """
    Load configuration from Streamlit's session state or initialize empty config.
    
    Returns:
        Dict containing configuration values
    """
    if 'config' not in st.session_state:
        st.session_state.config = {}
    return st.session_state.config

def save_config(config: Dict[str, Any]) -> None:
    """
    Save configuration to Streamlit's session state.
    
    Args:
        config: Dictionary containing configuration values
    """
    st.session_state.config = config
