import asyncio
from typing import TypedDict, Dict, Any, List
from anthropic import Anthropic
import json
from utils.llm import LLMHandler
from utils.vector_store import VectorStore
from utils.tools import LuminosoStats
import pandas as pd


class State(TypedDict):
    index: str
    preliminary_themes: Dict[str, Any]
    sample_df: Dict[str, Any]
    refined_themes: Dict[str, Any]
    tagged_df: Dict[str, Any]


async def generate_preliminary_themes(state: State, llm_handler: LLMHandler,
                                      vector_store: VectorStore,
                                      index: str) -> State:
    """Generate Themes and sentiments based on entire review set"""

    try:  #fetch reviews from MD
        reviews_df = vector_store.fetch_all_reviews(index).sample(
            750)  # token limit
        sample_df = reviews_df.sample(
            n=5).to_dict()  # for testing keyword extraction

        reviews_list = []

        for idx, row in reviews_df.iterrows():
            reviews_list.append(
                row['Text']
            )  # Changed extend to append since Text is a single value

        REVIEWS = '\n'.join(reviews_list)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch reviews: {str(e)}")
    try:
        response = llm_handler.anthropic.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=8192,
            temperature=0,
            system=
            """You are tasked with performing a thematic analysis on a set of reviews for a jewelry store. Your goal is to identify main themes and subthemes within these reviews. Return only valid JSON. Here are your instructions:

1. You will be provided with a set of reviews

2. Read through all the reviews carefully. As you read, take note of as many recurring topics, sentiments, or experiences mentioned by customers that you can.

3. Identify main themes:
   - Look for categories that encompass multiple reviews
   - These should be something that provides insights into customer experience that can be positive or negative emotions
   - Generate as many themes as you can, preferably more than 6, depending on the variety in the reviews

4. For each main theme, identify subthemes:
   - These should be sentiments within the main theme that shows the emotions

5. Organize your analysis as follows:

Structure for output:
{
  "themes": [
    {
      "theme": "",
      "subthemes": [
        {
          "name": "[Subtheme name]"
        }
        // Repeat for each subtheme
      ]
    }
    // Repeat for each main theme
  ]
}

        """,
            messages=[{
                "role":
                "user",
                "content":
                f"""
    <reviews>
    {REVIEWS}
    </reviews>

    Remember to focus on the content of the reviews and avoid making assumptions beyond what is explicitly stated. Your goal is to provide an objective analysis of the themes present in the customer feedback."""
            }])
        print(response.content[0].text)
        state["preliminary_themes"] = json.loads(response.content[0].text)
        state['sample_df'] = sample_df
        return state
    except Exception as e:
        raise RuntimeError(f"Themes-Sentiment generation failed: {str(e)}")


async def refine_themes(state: State, llm_handler: LLMHandler) -> State:
    """Combine Themes and sentiments to generate more meaningful themes"""
    try:
        response = llm_handler.anthropic.messages.create(
            model='claude-3-5-sonnet-20241022',
            max_tokens=4000,
            temperature=0,
            system=
            """I want you to go through these themes and subthemes. Combine these themes and subthemes into a new comprehensive yet mutually exclusive set of themes. The new themes are a combination of both the themes and subthemes. I only want you to give me the new themes and not any headings

Return only valid JSON in the following format:

{{"refined_themes": [theme1, theme2,.....]}}""",
            messages=[{
                "role":
                "user",
                "content":
                f"""Based on the reviews provided, here is my thematic analysis:
                {json.dumps(state['preliminary_themes'])}"""
            }])
        print(response.content[0].text)
        result = json.loads(response.content[0].text)
        if not isinstance(result, dict) or 'refined_themes' not in result:
            raise ValueError("Invalid refined themes format")
        state["refined_themes"] = result
        return state
    except Exception as e:
        raise RuntimeError(f"Refined Theme generation failed: {str(e)}")


async def generate_final_response(state: State,
                                  llm_handler: LLMHandler) -> State:
    """Generate final response combining all results."""
    try:
        if not isinstance(
                state['refined_themes'],
                dict) or 'refined_themes' not in state['refined_themes']:
            raise ValueError("Invalid refined themes format in state")

        themes_list = state['refined_themes']['refined_themes']
        if not isinstance(themes_list, list):
            raise ValueError("Refined themes must be a list")

        themes_text = '\n'.join(str(theme) for theme in themes_list)
        df = pd.DataFrame.from_dict(state['sample_df'])
        
        # Ensure Text column exists
        if 'Text' not in df.columns:
            df['Text'] = df.get('text', '')  # Try alternate column name
            
        # Initialize Theme_Subthemes column
        df['Theme_Subthemes'] = ''

        for idx, row in df.iterrows():
            review = row.get('Text', '')  # Safely get Text value

            keywords = llm_handler.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                system=
                f"""Analyze this review and output JSON with the following fields:
            Here are the combined and mutually exclusive themes:

            {themes_text}


            I will provide you with a sample review. Go through these themes and do the following task
            1. Categorize the sample review into the themes that above 
            2. Give me the specific keywords (two words or less each) directly from the sample review without changing or editing in any way that helped you in categorization. Do not repeat keywords that have very similar meaning. 
            3. Provide these in a JSON format in the following way -

            {{ 'theme' : [Keyword 1, Keyword 2...], 
            'theme' : [Keyword 1, Keyword 2...],...}}
                                    """,
                messages=[{
                    "role":
                    "user",
                    "content":
                    f"""
                Review: {review}

                Return only the JSON object, nothing else
                    """
                }])
            print(keywords)
            df.loc[idx, 'Theme_Subthemes'] = str(keywords.content[0].text)

        state["tagged_df"] = df.to_dict()
        return state
    except Exception as e:
        raise RuntimeError(f"Final review tagging failed: {str(e)}")


async def process_themes(index: str, llm_handler: LLMHandler,
                         vector_store: VectorStore) -> State:
    """Process a query through the workflow and return the final response."""
    try:
        # Initialize state
        state: State = {
            'index': index,
            'preliminary_themes': {},
            'sample_df': {},
            'refined_themes': {},
            'tagged_df': {}
        }

        # Step 1: Generate preliminary themes
        state = await generate_preliminary_themes(state, llm_handler,
                                                  vector_store, index)

        # Step 2: Refine themes
        state = await refine_themes(state, llm_handler)

        # Step 3: Generate final response
        state = await generate_final_response(state, llm_handler)

        return state

    except Exception as e:
        raise RuntimeError(f"Query processing failed: {str(e)}")
