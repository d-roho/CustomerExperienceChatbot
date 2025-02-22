import os
from typing import Dict, Any, List, Optional
import json
from luminoso_api import LuminosoClient
import pandas as pd
import datetime
import time
import asyncio
import utils.tools_funcs
import uuid 
import copy

def get_unix_time(month, year):
  # Create a datetime object for the first day of the given month and year
  dt = datetime.datetime(year, month, 1, 0, 0, 0)

  # Convert the datetime object to Unix time (seconds since January 1, 1970)
  unix_time = int(dt.timestamp())
  return unix_time
    

def filtering(filter):   
    filters = []
    filters_exist = 0
    for key in filter.keys():
        if key == 'themes':
            if filter[key]:
                drivers_exist = filter[key]
                themes = 1
            else:
                themes = 0
                drivers_exist = []
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
                ms = 1
        
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
    return filters, filters_exist, drivers_exist, themes


def sort_and_select(df: pd.DataFrame, cols: list, n: int) -> (pd.DataFrame):
    # Sort the DataFrame by two columns
    sorted_df = df.sort_values(by=cols, ascending=False)

    # Get the top N entries
    top_n = sorted_df.head(n)

    # Get the bottom N entries
    bottom_n = sorted_df.tail(n)

    # concatenate the top and bottom entries
    top_bottom_df = pd.concat([top_n, bottom_n], ignore_index=True)

    return top_bottom_df

def sort_and_select_sentiment(df: pd.DataFrame, cols: list, n: int) -> (pd.DataFrame):
    # Sort the DataFrame by two columns
    top_n = df.sort_values(by=[cols[0]], ascending=False).head(n)
    bottom_n = df.sort_values(by=[cols[1]], ascending=False).head(n)

    row_count = df.shape[0]
    mid_n =  df.sort_values(by=[cols[0]], ascending=False).iloc[(row_count//2)-(n//2):(row_count//2)+(n//2) ]

    concat_df = pd.concat([top_n, mid_n, bottom_n], ignore_index=True)
    return concat_df

def drivers_processing(result, api_start_time):
  if not isinstance(result, pd.DataFrame):
      df = pd.DataFrame(result)
  else:
      df = result
  df = df.drop(columns=[
      'texts', 'exact_term_ids',
      'excluded_term_ids', 'vectors', 'exact_match_count'
  ])
  try:
      df.drop(columns=['shared_concept_id', 'match_count'], inplace=True)
  except KeyError:
      pass

  try:
    df.drop(columns=['color'], inplace=True)
  except KeyError:
    pass

  # Select numeric columns (excluding 'name')
  numeric_cols = df.select_dtypes(include=['number']).columns

  # Normalize each numeric column
  for col in numeric_cols:
      min_val = df[col].min()
      max_val = df[col].max()
      if col in ['average_score', 'baseline']:
          # Handle constant columns by setting to 0.5
          pass
      elif col == 'impact':
          df[col] = df[col]
      else:
          df[col] = (df[col] - min_val) / (max_val - min_val)


    # order and filter

  filtered_df = df[df['relevance'] > 0.1]

  sort_and_filtered_df = sort_and_select(filtered_df, ['impact'], 25)

  column_mapping = {
      'name': 'Theme Name',
      'relevance': 'Normalized Relevance to Subset',
      # 'match_count': 'Matches Found in Subset',
      'impact': 'Impact on Score',
      'confidence': 'Impact Confidence Level',
      'average_score': 'Average Score',
      'baseline': 'Baseline Score'
  }

  # Rename the columns using the mapping
  df = sort_and_filtered_df.rename(columns=column_mapping)
  print(f"Cumulative processing time: {time.time() - api_start_time:.2f}s")
  print(df)
  return df


async def process_concept(concept):
  return {
      'Theme Name':
      concept['name'],
      'Proportion of Positive Mentions':
      concept['sentiment_share']['positive'],
      'Proportion of Neutral Mentions':
      concept['sentiment_share']['neutral'],
      'Proportion of Negative Mentions':
      concept['sentiment_share']['negative']
  }

def sentiments_processing(result, api_start_time):
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
      # if result['match_counts'].index(concept) % 10:
      #     print(concept)
      #     # print(result[str(result['match_counts'])[:100]])
  filter_count = result['filter_count']

  # Create DataFrame
  df = pd.DataFrame(rows)
  df['Proportion of Subset With Theme'] = df[
      'Proportion of Subset With Theme'] / filter_count


  filtered_df = df[df['Proportion of Subset With Theme'] > 0.1]
  final_df = sort_and_select_sentiment(filtered_df, ['Proportion of Positive Mentions', 'Proportion of Negative Mentions'], 15)
  

    
  return final_df

async def fetch_driver(theme, filters, filters_exist, client):
    if theme:
        concept = {"type": "concept_list", 'name': theme}
        result = await asyncio.to_thread(
            client.get,
            '/concepts/score_drivers/',
            score_field="Overall Rating",
            concept_selector=concept,
            filter=filters if filters_exist == 1 else None
        )
    else:
        result = await asyncio.to_thread(
            client.get,
            '/concepts/score_drivers/',
            score_field="Overall Rating",
            filter=filters if filters_exist == 1 else None,
            limit=None if theme else 50
        )
    return result

async def process_combinations(subset_combinations, filter, filters, filters_exist, client, api_start_time):
    drivers_dict = {}
    for combo in subset_combinations:
        current_filters = combo_breaker(combo, filter, filters)
        print(current_filters)

        # Fetch driver data
        result = await fetch_driver(None, current_filters, filters_exist, client)

        # Process the result
        df = pd.DataFrame(result)
        df = drivers_processing(df, api_start_time)
        drivers_dict[str(uuid.uuid4())] = {
            'df': df.to_dict(),
            'theme': 'All',
            'subset': copy.deepcopy(current_filters)
        }
        print(df)

    return drivers_dict



def combo_breaker(combo, filter, filters):
    idx = 0
    geo_list = ['states', 'location', 'cities']
    for sub in filter['subsets']:
        if sub in ['year_start', 'year_end', 'month_start', 'month_end', 'year']:
            for diction in  filters:
                if diction['name'] == 'Date Created':
                    diction['minimum'] = get_unix_time(1, combo[idx])
                    diction['maximum'] = get_unix_time(12, combo[idx])
                    idx+=1
        elif sub in ['states', 'location', 'cities']:
            mapper = {'states': 'State', 'location': 'Place Location', 'cities': 'City'}
            for diction in filters:
                if diction['name'] == mapper[sub]:
                    diction['values'] = [combo[idx]]
                    geo_list.remove(sub)
            for other_geo in geo_list:
                for diction in filters:
                    if diction['name'] == mapper[other_geo]:
                        filters.remove(diction)
                        
            idx+=1           
    return filters


async def get_sentiment(theme, filters, filters_exist, client):
    concept = {"type": "concept_list", 'name': theme}
    result = await asyncio.to_thread(lambda: client.get(
        '/concepts/sentiment/',
        concept_selector=concept,
        filter=filters if filters_exist == 1 else None))
    return result

