import os
from typing import Dict, Any, List, Optional
import json
from luminoso_api import LuminosoClient
import pandas as pd
import datetime
import time
import asyncio
import utils.tools_funcs
from utils.vector_funcs import subset_generator, hierarchy_upholder
import uuid
import copy


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

    async def fetch_drivers(self, client, filter):
        """Fetch drivers for given filters and concepts asynchronously."""
        try:
            start_time = time.time()
            filter, message = hierarchy_upholder(filter)
            if 'themes' in filter['subsets']:
                subset_filter = filter
                subset_filter['subsets'].remove('themes')
            else:
                subset_filter = filter
            subset_combinations, _ = subset_generator(subset_filter)

            filters, filters_exist, drivers_exist, themes = utils.tools_funcs.filtering(
                filter)

            filters_time = time.time() - start_time
            print(f"Filters preparation time: {filters_time:.2f}s")
            print(filter)
            print(f'{message}')
            print(f'subset_combinations: {subset_combinations}')
            counter = 0
            api_start_time = time.time()
            drivers_dict = {}
            if themes == 1:
                print(f'Themes: {drivers_exist}')

                if subset_combinations:
                    print('SUBSETTING')
                    for combo in subset_combinations:
                        filters = utils.tools_funcs.combo_breaker(
                            combo, filter, filters)

                        # Create tasks for all themes
                        tasks = [
                            utils.tools_funcs.fetch_driver(
                                theme, filters, filters_exist, client)
                            for theme in drivers_exist
                        ]

                        # Execute all tasks concurrently
                        results = await asyncio.gather(*tasks)

                        for idx, theme in enumerate(drivers_exist):
                            df = utils.tools_funcs.drivers_processing(
                                results[idx], api_start_time)
                            drivers_dict[str(uuid.uuid4())] = {
                                'subset': copy.deepcopy(filters),
                                'theme': theme,
                                'df': df.to_dict()
                            }

                else:
                    tasks = [
                        utils.tools_funcs.fetch_driver(theme, filters,
                                                       filters_exist, client)
                        for theme in drivers_exist
                    ]

                    # Execute all tasks concurrently
                    results = await asyncio.gather(*tasks)

                    for idx, theme in enumerate(drivers_exist):
                        print(results[idx])
                        df = utils.tools_funcs.drivers_processing(
                            results[idx], api_start_time)
                    drivers_dict[str(uuid.uuid4())] = {
                        'subset': copy.deepcopy(filters),
                        'theme': theme,
                        'df': df.to_dict()
                    }

            else:
                if subset_combinations:
                    print('SUBSETTING')
                    drivers_dict = await utils.tools_funcs.process_combinations(subset_combinations, filter, filters, filters_exist, client, api_start_time)
                
                else:
                    
                    task = [
                        utils.tools_funcs.fetch_driver(None, filters,
                                                       filters_exist, client)
                    ]
                    result = await asyncio.gather(*task)

                    df = pd.DataFrame(result[0])
                    df = utils.tools_funcs.drivers_processing(
                        df, api_start_time)
                    drivers_dict[str(uuid.uuid4())] = {
                        'subset': copy.deepcopy(filters),
                        'theme': 'All',
                        'df': df.to_dict()
                    }

                    print(df)
            total_time = time.time() - start_time
            print(f"Total fetch_drivers execution time: {total_time:.2f}s")
            return drivers_dict, total_time
        except Exception as e:
            raise RuntimeError(f"Failed to Fetch Drivers: {str(e)}")

    async def fetch_sentiment(self, client, filter):
        """Fetch sentiment for given filters and concepts asynchronously."""
        try:
            start_time = time.time()
            filter, message = hierarchy_upholder(filter)
            if 'themes' in filter['subsets']:
                subset_filter = filter
                subset_filter['subsets'].remove('themes')
            else:
                subset_filter = filter
            subset_combinations, _ = subset_generator(subset_filter)

            filters, filters_exist, sentiments_exist, themes = utils.tools_funcs.filtering(
                filter)

            filters_time = time.time() - start_time
            print(f"Filters preparation time: {filters_time:.2f}s")
            print(filters)
            counter = 0
            api_start_time = time.time()
            sentiments_dict = {}
            # themes = 0
            if themes == 1:
                print(sentiments_exist)
                if subset_combinations:
                    for combo in subset_combinations:
                        filters = utils.tools_funcs.combo_breaker(
                            combo, filter, filters)

                        # Create tasks for all themes
                        tasks = [
                            utils.tools_funcs.get_sentiment(
                                theme, filters, filters_exist, client)
                            for theme in sentiments_exist
                        ]

                        # Execute all tasks concurrently
                        results = await asyncio.gather(*tasks)

                        for idx, theme in enumerate(sentiments_exist):
                            df = utils.tools_funcs.sentiments_processing(
                                results[idx], api_start_time)

                            sentiments_dict[str(uuid.uuid4())] = {
                                'theme': theme,
                                'subset': copy.deepcopy(filters),
                                'df': df.to_dict()
                            }
                else:
                    # Create tasks for all themes
                    tasks = [
                        utils.tools_funcs.get_sentiment(
                            theme, filters, filters_exist, client)
                        for theme in sentiments_exist
                    ]

                    # Execute all tasks concurrently
                    results = await asyncio.gather(*tasks)

                    for idx, theme in enumerate(sentiments_exist):
                        df = utils.tools_funcs.sentiments_processing(
                            results[idx], api_start_time)

                    sentiments_dict[str(uuid.uuid4())] = {
                        'theme': theme,
                        'subset': copy.deepcopy(filters),
                        'df': df.to_dict()
                    }
            else:
                if subset_combinations:
                    print('SUBSETTING')
                    for idx, combo in enumerate(subset_combinations):
                        filters = utils.tools_funcs.combo_breaker(
                            combo, filter, filters)

                        result = await asyncio.to_thread(lambda: client.get(
                            '/concepts/sentiment/',
                            concept_selector={
                                "type": "top",
                                'limit': 50
                            },
                            filter=filters if filters_exist == 1 else None))

                        rows = []

                        # Create tasks for all concepts
                        tasks = [
                            utils.tools_funcs.process_concept(concept)
                            for concept in result['match_counts']
                        ]

                        # Execute all tasks concurrently
                        rows = await asyncio.gather(*tasks)
                        filter_count = result['filter_count']

                        # Create DataFrame
                        df = pd.DataFrame(rows)
                        print(df)
                        sentiments_dict[str(uuid.uuid4())] = {
                            'theme': 'All',
                            'subset': copy.deepcopy(filters),
                            'df': df.to_dict()
                        }                
                else:
                    result = await asyncio.to_thread(lambda: client.get(
                        '/concepts/sentiment/',
                        concept_selector={
                            "type": "top",
                            'limit': 50
                        },
                        filter=filters if filters_exist == 1 else None))

                    rows = []

                    # Create tasks for all concepts
                    tasks = [
                        utils.tools_funcs.process_concept(concept)
                        for concept in result['match_counts']
                    ]

                    # Execute all tasks concurrently
                    rows = await asyncio.gather(*tasks)
                    filter_count = result['filter_count']

                    # Create DataFrame
                    df = pd.DataFrame(rows)
                    print(df)
                    sentiments_dict[str(uuid.uuid4())] = {
                        'theme': 'All',
                        'subset': copy.deepcopy(filters),
                        'df': df.to_dict()
                    }
            total_time = time.time() - start_time
            print(f"Total fetch_sentiment execution time: {total_time:.2f}s")
            return sentiments_dict, total_time
        except Exception as e:
            raise RuntimeError(f"Failed to Fetch Sentiment: {str(e)}")
