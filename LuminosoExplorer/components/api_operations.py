import streamlit as st
import pandas as pd
from typing import Any


def render_api_operations(client: Any):
    st.header("API Operations")

    operation = st.selectbox(
        "Select Operation",
        ["List Projects", "Project Details", "Get Drivers", "Get Sentiment", "Combine"])

    if operation == "List Projects":
        if st.button("Fetch Projects"):
            try:
                with st.spinner("Fetching projects..."):
                    projects = client.get()
                    df = pd.DataFrame(projects)
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to fetch projects: {str(e)}")

    elif operation == "Project Details":
            try:
                with st.spinner("Fetching project details..."):
                    details = client.get()
                    st.json(details)
            except Exception as e:
                st.error(f"Failed to fetch project details: {str(e)}")

    elif operation == "Get Drivers":

        if st.button("Download"):
            try:
                with st.spinner("Downlaoding..."):
                    concept = {
                        "type":
                        "concept_list",
                        'name':
                        'Shopping Experience'
                    }
                    filter = [{
                           "name": "Overall Rating",
                           "maximum": 5
                       }]
                    result = client.get('/concepts/score_drivers/', score_field="Overall Rating", concept_selector=concept, filter=filter
                                        )

                st.success("Downloaded!")
                df = pd.DataFrame(result)
                df = df.drop(columns=['shared_concept_id', 'color', 'texts', 'exact_term_ids', 'excluded_term_ids', 'vectors', 'exact_match_count'])
                # Select numeric columns (excluding 'name')
                numeric_cols = df.select_dtypes(include=['number']).columns

                # Normalize each numeric column
                for col in numeric_cols:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if col in ['average_score', 'baseline']:
                        # Handle constant columns by setting to 0.5
                        df[col] = df[col]/5.0
                    elif col == 'impact':
                        df[col] = df[col]
                    else:
                        df[col] = (df[col] - min_val) / (max_val - min_val)

                column_mapping = {
                    'name': 'Theme Name',
                    'relevance': 'Normalized Relevance to Subset',
                    'match_count': 'Matches Found in Subset',
                    'impact': 'Impact on Score',
                    'confidence': 'Impact Confidence Level',
                    'average_score': 'Average Score',
                    'baseline': 'Baseline Score'
                }

                # Rename the columns using the mapping
                df = df.rename(columns=column_mapping)
                st.dataframe(df)

            except Exception as e:
                st.error(f"Failed to download: {str(e)}")
        else:
            st.warning("Please enter a project name")

    elif operation == "Get Sentiment":

        if st.button("Download"):
            try:
                with st.spinner("Downlaoding..."):
                    filter=[{
                        "name": "Overall Rating",
                        "maximum": 5
                    }]
                    concept = {
                        "type":
                        "concept_list",
                        "name":
                        'Helios Themes'
                    }
                    result = client.get('concepts/sentiment/',
                                        concept_selector=concept, filter=filter
                                        
                                       )

                st.success("Downloaded!")
                rows = []
                for concept in result['match_counts']:
                    row = {
                        'Theme Name': concept['name'],
                        'Proportion of Subset With Theme': concept['match_count'],
                        'Proportion of Positive Mentions': concept['sentiment_share']['positive'],
                        'Proportion of Neutral Mentions': concept['sentiment_share']['neutral'],
                        'Proportion of Negative Mentions': concept['sentiment_share']['negative']
                    }
                    rows.append(row)
                    filter_count = result['filter_count'] 

                # Create DataFrame
                df = pd.DataFrame(rows)
                df['Proportion of Subset With Theme'] = df['Proportion of Subset With Theme']/filter_count 

                st.dataframe(df)
            except Exception as e:
                st.error(f"Failed to download: {str(e)}")
        else:
            st.warning("Please enter a project name")


    elif operation == "Combine":
    
        if st.button("Combine"):
            try:
                with st.spinner("Working..."):

                    result = client.get('concept_lists/', name= 'Test Themes', concepts = [{'name': 'Aesthetic Design & Visual Appeal'}])
                    st.code(str(result[:100]))
                st.success("Done!")
            except Exception as e:
                st.error(f"Failed to Create: {str(e)}")
        else:
            st.warning("Please enter a project name")