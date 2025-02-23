from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from anthropic import Anthropic
from utils.db import MotherDuckStore  # New import
import datetime
import asyncio
import itertools


def hierarchy_upholder(filters):
  message = 'Removed lower level subsets: '
  if 'states' in filters['subsets']:
    if 'cities' in filters['subsets']:
      filters['subsets'].remove('cities')
      message += 'cities, '
    if 'location' in filters['subsets']:
      filters['subsets'].remove('location')
      message += 'location,'
  if 'cities' in filters['subsets']:
    if 'location' in filters['subsets']:
      filters['subsets'].remove('location')
      message += 'location,'
  return filters, message


def subset_generator(filters):
  # Handle year ranges more robustly
  year_start = filters.get('year_start', [2019])[0] if filters.get('year_start') else 2019
  year_end = filters.get('year_end', [2025])[0] if filters.get('year_end') else 2025
  year_range = year_end - year_start + 1
  years = [year_start + i for i in range(year_range)]
  filters['years'] = years

  subset_options_merged = []

  date_types = ['year_start', 'year_end', 'month_start', 'month_end', 'year']
  subset_options_1 = [
      filters[subset] for subset in filters['subsets']
      if subset not in date_types
  ]
  subset_options_merged.extend(subset_options_1)

  has_date = set(date_types) & set(filters['subsets'])
  if has_date:
    subset_options_merged.append(years)

  if subset_options_merged:
    subset_combinations = list(itertools.product(*subset_options_merged))
  else:
    subset_combinations = [()]
  print("Subset combinations:", subset_combinations)
  return subset_combinations, has_date


async def process_combination(self, query_embedding, top_k, index_name, filters: dict, combo_idx: int, combo: tuple,
                              has_date: bool):

  FIELD_MAPPING = {
      'cities': 'city',
      'states': 'state',
      'month_start': 'date_month',
      'year_start': 'date_year',
      'month_end': 'date_month',
      'year_end': 'date_year',
      'rating_min': 'rating',
      'rating_max': 'rating',
      'location': 'location'
  }

  filter_query = {}

  rating_conditions = {}
  if filters.get('rating_min'):
    rating_conditions['$gte'] = filters['rating_min'][0]
  if filters.get('rating_max'):
    rating_conditions['$lte'] = filters['rating_max'][0]
  if rating_conditions:
    filter_query['rating'] = rating_conditions

  def get_unix_time(month, year):
    dt = datetime.datetime(year, month, 1, 0, 0, 0)
    unix_time = int(dt.timestamp())
    return unix_time

  if has_date:
    filter_query['date_year'] = {'$eq': combo[-1]}

  else:
    date_conditions = {}

    if filters.get('year_start'):
      if filters.get('month_start'):
        start_unix = get_unix_time(filters['month_start'][0],
                                   filters['year_start'][0])
      else:
        start_unix = get_unix_time(1, filters['year_start'][0])
      date_conditions['$gte'] = start_unix

    if filters.get('year_end'):
      if filters.get('month_end'):
        end_unix = get_unix_time(filters['month_end'][0],
                                 filters['year_end'][0])
      else:
        end_unix = get_unix_time(12, filters['year_end'][0])
      date_conditions['$lte'] = end_unix

    if date_conditions:
      filter_query['date_unix'] = date_conditions

    for key in filters:
      if key in [
          'rating_min', 'rating_max', 'subsets', 'month_start', 'year_start',
          'month_end', 'year_end'
      ]:
        continue

      values = filters[key]
      if not values:
        continue

      if key.lower() == 'themes':
        if 'themes' in filters['subsets']:
          idx = filters['subsets'].index('themes')
          filter_query[combo[idx]] = {'$exists': True}

        else:
          filter_query['$or'] = [{
              theme: {
                  '$exists': True
              }
          } for theme in filters[key]]
      if key in FIELD_MAPPING:
        mongo_field = FIELD_MAPPING[key]
        if key in filters['subsets']:
          idx = filters['subsets'].index(key)
          filter_query[mongo_field] = {'$eq': combo[idx]}
        else:
          filter_query[mongo_field] = {'$in': values}

  print(filter_query)

  results = self.index.query(vector=query_embedding,
                             top_k=top_k,
                             filter=filter_query,
                             include_metadata=True)

  print(f"Successfully retrieved {len(results['matches'])} results")

  processed_results = []
  for match in results.matches:
    stored_data = self.db.get_chunk(match.id, index_name)
    if stored_data:
      processed_results.append({
          'text': stored_data['text'],
          'metadata': stored_data['metadata'],
          'header': match.metadata['header'],
          'score': match.score
      })
  subset_info = {}
  for idx, sub in enumerate(filters['subsets']):
    subset_info[sub] = combo[idx]

  return {
      'combo_idx': combo_idx,
      'subset_info': subset_info,
      'processed_results': processed_results
  }
