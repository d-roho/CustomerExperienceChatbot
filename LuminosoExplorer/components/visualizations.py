import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Any

def render_visualizations(client: Any):
    st.header("Data Visualizations")
    
    project_id = st.text_input("Enter Project ID for Visualization")
    
    if not project_id:
        st.info("Please enter a project ID to view visualizations")
        return
        
    try:
        # Fetch project data
        with st.spinner("Fetching project data..."):
            project_data = client.get(project_id)
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Concept Cloud", "Timeline Analysis"])
            
            with tab1:
                render_concept_cloud(project_data)
                
            with tab2:
                render_timeline_analysis(project_data)
                
    except Exception as e:
        st.error(f"Failed to fetch project data: {str(e)}")

def render_concept_cloud(project_data: dict):
    st.subheader("Concept Cloud")
    
    # Convert concepts to DataFrame
    if 'concepts' in project_data:
        df = pd.DataFrame(project_data['concepts'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['relevance'],
            y=df['frequency'],
            mode='markers+text',
            text=df['name'],
            textposition='top center',
            marker=dict(
                size=df['relevance'] * 50,
                color=df['frequency'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="Concept Relevance vs Frequency",
            xaxis_title="Relevance",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        st.plotly_chart(fig)
    else:
        st.info("No concept data available for visualization")

def render_timeline_analysis(project_data: dict):
    st.subheader("Timeline Analysis")
    
    if 'timeline_data' in project_data:
        df = pd.DataFrame(project_data['timeline_data'])
        
        fig = px.line(
            df,
            x='timestamp',
            y='value',
            color='metric',
            title="Metrics Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            legend_title="Metric"
        )
        
        st.plotly_chart(fig)
    else:
        st.info("No timeline data available for visualization")
