import os
from typing import Dict, Any, List, Optional
import json
from luminoso_api import LuminosoClient
import pandas as pd
import datetime


class LuminosoStats:

    def __init__(self):
        """Initialize LUMINOSO API connection."""
        self.luminoso_token = os.environ.get('LUMINOSO_TOKEN')
        if not self.luminoso_token:
            raise ValueError("LUMINOSO_TOKEN environment variable is required")
        self.luminoso_url = os.environ.get('LUMINOSO_URL')
        if not self.luminoso_url:
            raise ValueError("LUMINOSO_URL environment variable is required")

    def initialize_client(self) -> LuminosoClient:
        """     
        Returns:
            LuminosoClient: Initialized API client

        Raises:
            ValueError: If API credentials are missing
            Exception: For other initialization errors
        """
        try:
            client = LuminosoClient.connect(url=self.luminoso_url,
                                            token=self.luminoso_token)
            return client
        except Exception as e:
            raise Exception(
                f"Failed to initialize Luminoso API client: {str(e)}")

    def fetch_drivers(self, client, filter):
        """Fetch drivers for given filters and concepts."""
        try:

            def get_unix_time(month, year):
                # Create a datetime object for the first day of the given month and year
                dt = datetime.datetime(year, month, 1, 0, 0, 0)

                # Convert the datetime object to Unix time (seconds since January 1, 1970)
                unix_time = int(dt.timestamp())
                return unix_time

            filters = []
            filters_exist = 0
            for key in filter.keys():
                if key == 'themes':
                    if filter[key]:
                        drivers_exist = filter[key]
                        themes = 1
                    else:
                        themes = 0
                if key == 'rating_min':
                    try:
                        rat_min = filter[key][0]
                    except:
                        rat_min = 0

                    try:
                        rat_max = filter['rating_max'][0]
                    except:
                        rat_max = 5

                    dict = {
                        "name": 'Overall Rating',
                        "minimum": rat_min,
                        "maximum": rat_max
                    }
                    print(dict)
                    filters.append(dict)
                    filters_exist = 1
                if key == 'cities':
                    if filter[key]:
                        dict = {"name": 'City', "values": filter[key]}
                        filters.append(dict)
                        filters_exist = 1

                if key == 'location':
                    if filter[key]:
                        dict = {
                            "name": 'Place Location',
                            "values": filter[key]
                        }
                        filters.append(dict)
                        filters_exist = 1

                if key == 'states':
                    if filter[key]:
                        dict = {"name": 'State', "values": filter[key]}
                        filters.append(dict)
                        filters_exist = 1

                if key == 'month_start':
                    try:
                        ms = filter[key][0]
                    except:
                        ms = 0

                    try:
                        ys = filter['year_start'][0]
                    except:
                        ys = 2019

                    try:
                        me = filter['month'][0]
                    except:
                        me = 12

                    try:
                        ye = filter['year_end'][0]
                    except:
                        ye = 2025

                    start_date = get_unix_time(ms, ys)
                    end_date = get_unix_time(me, ye)
                    dict = {
                        "name": 'Date Created',
                        "minimum": start_date,
                        "maximum": end_date
                    }
                    filters.append(dict)
                    filters_exist = 1

            print(filters)
            counter = 0
            themes = 0
            if themes == 1:
                print(drivers_exist)
                for theme in drivers_exist:
                    concept = {"type": "concept_list", 'name': theme}
                    if filters_exist == 1:
                        result = client.get('/concepts/score_drivers/',
                                            score_field="Overall Rating",
                                            concept_selector=concept,
                                            filter=filters)

                    else:
                        result = client.get(
                            '/concepts/score_drivers/',
                            score_field="Overall Rating",
                            concept_selector=concept,
                            # filter=filter
                        )

                    df = pd.DataFrame(result)
                    df = df.drop(columns=[
                        'color', 'texts', 'exact_term_ids',
                        'excluded_term_ids', 'vectors', 'exact_match_count'
                    ])
                    try:
                        df.drop(columns=['shared_concept_id'], inplace=True)
                    except:
                        continue
                    # Select numeric columns (excluding 'name')
                    numeric_cols = df.select_dtypes(include=['number']).columns

                    # Normalize each numeric column
                    for col in numeric_cols:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if col in ['average_score', 'baseline']:
                            # Handle constant columns by setting to 0.5
                            df[col] = df[col] / 5.0
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
                    print(df)
                    if counter == 0:
                        df_deep_copy = df.copy(deep=True)
                    else:
                        df_to_merge = df.copy(deep=True)
                        df_deep_copy = pd.concat([df_deep_copy, df_to_merge])
                    counter += 1
                    print(len(df_deep_copy))

            else:
                if filters_exist == 1:
                    result = client.get('/concepts/score_drivers/',
                                        score_field="Overall Rating",
                                        limit=50,
                                        filter=filters)

                else:
                    result = client.get(
                        '/concepts/score_drivers/',
                        score_field="Overall Rating",
                        limit=50,
                        # filter=filter
                    )

                df = pd.DataFrame(result)
                df = df.drop(columns=[
                    'texts', 'exact_term_ids', 'excluded_term_ids', 'vectors',
                    'exact_match_count'
                ])
                # Select numeric columns (excluding 'name')
                numeric_cols = df.select_dtypes(include=['number']).columns

                # Normalize each numeric column
                for col in numeric_cols:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if col in ['average_score', 'baseline']:
                        # Handle constant columns by setting to 0.5
                        df[col] = df[col] / 5.0
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
                print(df)
                df_deep_copy = df.copy(deep=True)

            return df_deep_copy
        except Exception as e:
            raise RuntimeError(f"Failed to Fetch Drivers: {str(e)}")

    def fetch_sentiment(self, client, filter):
        """Fetch sentiment for given filters and concepts."""
        try:

            def get_unix_time(month, year):
                # Create a datetime object for the first day of the given month and year
                dt = datetime.datetime(year, month, 1, 0, 0, 0)

                # Convert the datetime object to Unix time (seconds since January 1, 1970)
                unix_time = int(dt.timestamp())
                return unix_time

            filters = []
            filters_exist = 0
            for key in filter.keys():
                if key == 'themes':
                    if filter[key]:
                        sentiments_exist = filter[key]
                        themes = 1
                    else:
                        themes = 0
                if key == 'rating_min':
                    try:
                        rat_min = filter[key][0]
                    except:
                        rat_min = 0

                    try:
                        rat_max = filter['rating_max'][0]
                    except:
                        rat_max = 5

                    dict = {
                        "name": 'Overall Rating',
                        "minimum": rat_min,
                        "maximum": rat_max
                    }
                    print(dict)
                    filters.append(dict)
                    filters_exist = 1
                if key == 'cities':
                    if filter[key]:
                        dict = {"name": 'City', "values": filter[key]}
                        filters.append(dict)
                        filters_exist = 1

                if key == 'location':
                    if filter[key]:
                        dict = {
                            "name": 'Place Location',
                            "values": filter[key]
                        }
                        filters.append(dict)
                        filters_exist = 1

                if key == 'states':
                    if filter[key]:
                        dict = {"name": 'State', "values": filter[key]}
                        filters.append(dict)
                        filters_exist = 1

                if key == 'month_start':
                    try:
                        ms = filter[key][0]
                    except:
                        ms = 0

                    try:
                        ys = filter['year_start'][0]
                    except:
                        ys = 2019

                    try:
                        me = filter['month'][0]
                    except:
                        me = 12

                    try:
                        ye = filter['year_end'][0]
                    except:
                        ye = 2025

                    start_date = get_unix_time(ms, ys)
                    end_date = get_unix_time(me, ye)
                    dict = {
                        "name": 'Date Created',
                        "minimum": start_date,
                        "maximum": end_date
                    }
                    filters.append(dict)
                    filters_exist = 1

            print(filters)
            counter = 0
            # themes = 0
            if themes == 1:
                print(sentiments_exist)
                for theme in sentiments_exist:
                    concept = {"type": "concept_list", 'name': theme}
                    if filters_exist == 1:
                        result = client.get('/concepts/sentiment/',
                                            concept_selector=concept,
                                            filter=filters)

                    else:
                        result = client.get(
                            '/concepts/sentiment/',
                            concept_selector=concept,
                            # filter=filter
                        )

                    rows = []
                    for concept in result['match_counts']:
                        row = {
                            'Theme Name':
                            concept['name'],
                            'Proportion of Subset With Theme':
                            concept['match_count'],
                            'Proportion of Positive Mentions':
                            concept['sentiment_share']['positive'],
                            'Proportion of Neutral Mentions':
                            concept['sentiment_share']['neutral'],
                            'Proportion of Negative Mentions':
                            concept['sentiment_share']['negative']
                        }
                        rows.append(row)
                        filter_count = result['filter_count']

                # Create DataFrame
                df = pd.DataFrame(rows)
                df['Proportion of Subset With Theme'] = df[
                    'Proportion of Subset With Theme'] / filter_count
                if counter == 0:
                    df_deep_copy = df.copy(deep=True)
                else:
                    df_to_merge = df.copy(deep=True)
                    df_deep_copy = pd.concat([df_deep_copy, df_to_merge])
                counter += 1
                print(len(df_deep_copy))
                df_deep_copy = df_deep_copy.sort_values(
                    by='Proportion of Subset With Theme', ascending=False)
                df_deep_copy = df_deep_copy.head(50)

            else:
                if filters_exist == 1:
                    result = client.get('/concepts/sentiment/',
                                        concept_selector={
                                            "type": "top",
                                            'limit': 50
                                        },
                                        filter=filters)

                else:
                    result = client.get(
                        '/concepts/sentiment/',
                        concept_selector={
                            "type": "top",
                            'limit': 50
                        },
                        # filter=filters
                    )

                rows = []
                for concept in result['match_counts']:
                    row = {
                        'Theme Name':
                        concept['name'],
                        # 'Proportion of Subset With Theme': concept['match_count'],
                        'Proportion of Positive Mentions':
                        concept['sentiment_share']['positive'],
                        'Proportion of Neutral Mentions':
                        concept['sentiment_share']['neutral'],
                        'Proportion of Negative Mentions':
                        concept['sentiment_share']['negative']
                    }
                    rows.append(row)
                filter_count = result['filter_count']

                # Create DataFrame
                df = pd.DataFrame(rows)
                # df['Proportion of Subset With Theme'] = df['Proportion of Subset With Theme']/filter_count
                print(df)
                df_deep_copy = df.copy(deep=True)

            return df_deep_copy
        except Exception as e:
            raise RuntimeError(f"Failed to Fetch Sentiment: {str(e)}")
