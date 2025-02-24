import asyncio
from typing import TypedDict, Dict, Any, List
from anthropic import Anthropic
import json
from utils.llm import LLMHandler
from utils.vector_store import VectorStore
from utils.tools import LuminosoStats
import time

possible_subsets  = ["cities", "rating_min", "rating_max", "month_start", "year_start", "month_end", "year_end", "location", "states", "themes"]
generate_filters_prompt = """Extract metadata from the query in the following format. Return only valid JSON

        {'cities': [] ,
        'rating_min': [],
        'rating_max': [],
        'month_start': [],
        'year_start': [],
        'month_end': [],
        'year_end': [],
        'location': [],
        'states': [],
        'themes':  []
        'subsets': [] }

        Only choose from the possible values provided below. Leave the value blank if it is not specified in the query.  

        'cities': [Austin, Bellevue, Bethesda, Boston, Brooklyn, Chestnut Hill, Chicago, Denver, Houston, Los Angeles, Miami, Montreal, Nashville, New York, North York, Philadelphia, San Diego, Seattle, Short Hills, Skokie, Toronto, Vancouver, West Vancouver] ,
        'rating_min': [1,2,3,4,5],
        'rating_max': [1,2,3,4,5],
        'month_start': [1,2,3,4,5,6,7,8,9,10,11,12],
        'year_start': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'month_end':  [1,2,3,4,5,6,7,8,9,10,11,12],
        'year_end': [2019, 2020, 2021, 2022, 2023, 2024, 2025],
        'location': [43 Spring St, New York, NY | 8404 Melrose Ave, Los Angeles, CA | 11700 Domain Blvd Suite 126, Austin, TX | 2166 W 4th Ave, Vancouver, BC, Canada | 126 Newbury St, Boston, MA | 1410 Peel St, Montreal, Quebec, Canada | 3401 Dufferin St, North York, ON, Canada | 940 W Randolph St, Chicago, IL, United States | 888 Westheimer Rd Suite 158, Houston, TX | 4545 La Jolla Village Dr Suite C-12, San Diego, CA | 2621 NE University Village St, Seattle, WA | 107 N 6th St, Brooklyn, NY | 144 5th Ave, New York, NY | 1525 Walnut St, Philadelphia, PA | 7247 Woodmont Ave, Bethesda, MD | 64 Ossington Ave, Toronto, ON, Canada | 2803 12th Ave S, Nashville, TN | 219 NW 25th St, Miami, FL | 925 Main St Unit H3, West Vancouver, BC, Canada | 124 Bellevue Square Unit L124, Bellevue, WA | 1200 Morris Tpke, Short Hills, NJ, United States | 3000 E 1st Ave #144, Denver, CO | 4999 Old Orchard Shopping Ctr Suite B34, Skokie, IL, United States | 737 Dunsmuir St, Vancouver, BC, Canada | 27 Boylston St, Chestnut Hill, MA]
        'states': [NY , CA , TX , BC , MA , QC , ON , IL , WA , PA , MD , TN , FL , NJ , CO],
        'themes':  [Exceptional Customer Service & Support , Poor Service & Long Wait Times , Product Durability & Quality Issues , Aesthetic Design & Visual Appeal , Professional Piercing Services & Environment, Store Ambiance & Try-On Experience , Price & Policy Transparency , Store Organization & Product Selection , Complex Returns & Warranty Handling , Communication & Policy Consistency , Value & Price-Quality Assessment , Affordable Luxury & Investment Value , Online Shopping Experience , Inventory & Cross-Channel Integration]
        'subsets': Go through the given query and find out all the fields that can be used for comparison with other subsets. These should be selected from the earlier used fields and should be highly relevant. You can give multiple fields if necessary. Use from these (cities, rating_min, rating_max, month_start, year_start, month_end, year_end, location, states, themes)

        Current Date in February 2025"""

drivers_summary_prompt = """You are an expert customer experience analyst that gathers insights from aggregate rating drivers data. Your task is to analyze the following data and make a thorough summary to be used to understand the overall customer experience. Align your summary with the User Query so as to cover everything that may be required for answering the query. Assume the data provided is for the date and locations etc. asked by the user

                In your analysis, Be sure to include statistics to back up your analysis. Here is an explanation of the data:

                Dataset - A collection of themes (aspects of customer experience) and their impact on customer ratings. 

                Theme Name: Name of the aspect
                Normalized Relevance to Subset: The proportion of the total number of customers that this theme is relevant to (0-1). This is an indicator of pervasiveness of the theme.
                Impact on Score: effect of this theme on the customer ratings (-1 to +1)
                Impact Confidence Level: The confidence level of the impact on score. This is an indicator of the reliability of the impact on score. (0-1)
                Average Score: The average rating of the customer reviews that include this theme. (0-5 stars)
                Baseline Score: The average rating of all customers.(0-5 stars)            
"""
sentiment_summary_prompt = """You are an expert customer experience analyst that gathers insights from aggregate sentiment data. Your task is to analyze the following data and make a thorough summary to be used to understand the overall customer experience. Align your summary with the User Query so as to cover everything that may be required for answering the query. Assume the data provided is for the date and locations etc. asked by the user

                In your analysis, Be sure to include statistics to back up your analysis. Here is an explanation of the data:

                Dataset - A collection of themes (aspects of customer experience) and customer sentiment around those themes. 

                Theme Name: Name of the aspect
                Proportion of Subset With Theme: The proportion of the total reviews that include this theme (0-1). This is an indicator of pervasiveness of the theme.)
                Proportion of Positive Mentions: The proportion theme mentions made with a positive sentiment (0-1)
                Proportion of Neutral Mentions: The proportion theme mentions made with a neutral sentiment (0-1)
                Proportion of Negative Mentions: The proportion theme mentions made with a neutral sentiment (0-1)
"""

combined_summary_prompt = """You are an expert customer experience analyst that gathers insights from aggregate data. Your task is to analyze the following data and make a thorough summary to be used to understand the overall customer experience. Align your summary with the User Query so as to cover everything that may be required for answering the query. Assume the data provided is for the date and locations etc. asked by the user

                In your analysis, Be sure to include statistics to back up your analysis. Here is an explanation of the data:

                Drivers Dataset - A collection of themes (aspects of customer experience) and their impact on customer ratings. 
                
                Theme Name: Name of the aspect
                Normalized Relevance to Subset: The proportion of the total number of customers that this theme is relevant to (0-1). This is an indicator of pervasiveness of the theme.
                Impact on Score: effect of this theme on the customer ratings (-1 to +1)
                Impact Confidence Level: The confidence level of the impact on score. This is an indicator of the reliability of the impact on score. (0-1)
                Average Score: The average rating of the customer reviews that include this theme. (0-5 stars)
                Baseline Score: The average rating of all customers.(0-5 stars)            
                
                Sentiment Dataset - A collection of themes (aspects of customer experience) and customer sentiment around those themes. 

                Theme Name: Name of the aspect
                Proportion of Subset With Theme: The proportion of the total reviews that include this theme (0-1). This is an indicator of pervasiveness of the theme.)
                Proportion of Positive Mentions: The proportion theme mentions made with a positive sentiment (0-1)
                Proportion of Neutral Mentions: The proportion theme mentions made with a neutral sentiment (0-1)
                Proportion of Negative Mentions: The proportion theme mentions made with a neutral sentiment (0-1)
                
                
                Use impact and sentiment metrics equally, mention both of them in your analysis as often as possible."""


reviews_summary_prompt ="""You are a helpful customer experience analysis expert that provides insights from raw customer reviews. Provide a comprehensive analysis of the customer experience based on the USER QUERY and the raw reviews. Be sure to direct quotes from reviews to back up your analysis everywhere possible.   

Bullet points should be used wherever necessary
Reference the source text wherever using them to answer the question
include Subset and Review Number in the analysis for easy reference 
"""

final_answer_prompt = """You are a helpful customer experience analysis expert that provides insights from aggregate data summaries and customer reviews. Combine the data summaries and customer reviews to provide a comprehensive analysis of the customer experience. Be sure to include statistics and direct quotes from reviews to back up your analysis everywhere possible.   

            Provide a well-structured response that includes:
            A basic line to summarize the question and give a start to the answer. Keep it neutral and informative

            Bullet points should be used wherever necessary
            Reference the source wherever using them to answer the question, including subset and review number
            Use impact and sentiment metrics equally, mention both of them in your analysis as often as possible with appropriate units

            Summary of the answer in 2-3 lines emphasizing the most important points.
            Suggested actions based on findings and query if necessary
            Provide potential follow up query to answer if necessary"""

final_answer_prompt_concise = """You are a helpful customer experience analysis expert that provides insights from existing reports on customer experience. There are two types of reports:
    Aggregate Data Analysis Reports: Provides a comprehensive analysis of the customer experience based on the raw data on Ratings Drivers and Customer Sentiment directly related to USER QUERY.
    Reviews Analysis Reports: Provides a comprehensive analysis of the customer experience based on customer reviews related to USER QUERY.
    
Combine the reports to provide a concise analysis of the customer experience. Be sure to only include what is already in the reports, only condensing the existing report info to answer the USER QUERY as efficiently as possible.   

Provide a well-structured response that includes:

Bullet points should be used wherever necessary
References quotes from reviews wherever possible, including subset and review number in parenteses
Use impact and sentiment metrics equally, mention both of them in your analysis as often as possible with appropriate units
Do not include introductions or summaries, stick to the main points.  
do not include [review] [aggregate data] or other such tags in the response, only the actual information from those sources 
"""

def context_generator(context):
    # for key in context.keys():
    #     value = context[key]
    #     title = f"Subset Info: \n {json.dumps(value['subset_info'], indent=2)}"
    #     pc_metadata = value['processed_results']['pc_metadata']
    #     useful_metadata = ['string_City', 'score_Overall_Rating', 'string_Place_Location', 'string_State', 'themes', 'score_Overall_Rating', 'date_Date_Created']
    #     mapper = {"cities":"string_City", "location":"string_Place_Location", "states":"string_State", "themes":"themes"}
    #     for sub in value['subset_info']:
    #         try:
    #             useful_metadata.remove(mapper[sub])
    #         except:
    #             continue

    #     header_attr = []
    #     if 'string_Place_Location' in useful_metadata:
    #         header_attr.append(f'Location - {pc_metadata["string_Place_Location"]}')
    #     if 'date_Date_Created' in useful_metadata:
    #         header_attr.append(f'Date (MM/YYYY) - {pc_metadata["date_Date_Created"]}')
    #     if 'string_Place_Location' in useful_metadata:
    #         header_attr.append(f'Location - {pc_metadata["string_Place_Location"]}')
    #     if 'string_Place_Location' in useful_metadata:
    #         header_attr.append(f'Location - {pc_metadata["string_Place_Location"]}')
    #     if 'string_Place_Location' in useful_metadata:
    #         header_attr.append(f'Location - {pc_metadata["string_Place_Location"]}')
            
    #     header =  f" - {metadata['location']}, Date (MM/YYYY) - {metadata['date_month']}/{metadata['date_year']}, Rating - {metadata['rating']}/5.0, Upvotes {metadata['likes']}\nThemes - {row_themes}"

    #     rev_header =  
        
    #     for 

    #     context_text = "\n".join(
    #         [
    #             f"Review {idx} (Retriever Score: {c['score']})  \n - Text: {c['text']}\n\n"
    #             if c['header'] not in cumulative_reviews
    #             else f"Review {idx} \n {c['text']}"
    #             for idx, c in enumerate(value['processed_results'])
    #         ]
    #     )

    context_list = []
    cumulative_reviews = []
    for key in context.keys():
        value = context[key]
        title = f"Subset {key + 1} Info: \n {json.dumps(value['subset_info'], indent=2)}"

        context_text = "\n".join(
            [
                f"Review {idx + 1} (Retriever Score: {c['score']}) \nMetadata: {c['header']} \n - Text: {c['text']}\n\n"
                if c['header'] not in cumulative_reviews
                else f"Review {idx + 1} \n {c['text']}"
                for idx, c in enumerate(value['processed_results'])
            ]
        )


        context_list.append(f"{title}\n\n {context_text} \n\n END OF SUBSET {key} \n\n")    
        cumulative_reviews.extend([c['header'] for idx, c in enumerate(value['processed_results'])]) #remove duplicate reviews       
    context_text = "\n\n".join(context_list)

    return context_text

