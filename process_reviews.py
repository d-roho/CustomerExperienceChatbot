import pandas as pd
import anthropic
import json
from datetime import datetime
import os
import sys
import itertools
import time


def process_review_with_llm(client, THEMES_LIST, review_text):
    """Process a single review through Claude and get structured output"""
    try:
        message = client.messages.create(
            model=
            "claude-3-5-sonnet-20241022",  # newest Anthropic model as of Oct 22, 2024
            max_tokens=1000,
            temperature=0,
            system=
            "You are a helpful assistant that analyzes customer reviews. Output only valid JSON.",
            messages=[{
                "role":
                "user",
                "content":
                f"""Analyze this review and output JSON with the following fields:
            Here are the combined and mutually exclusive themes:
            
            {THEMES_LIST}
            
            
            I will provide you with a sample review. Go through these themes and do the following task
            1. Categorize the sample review into the themes that above 
            2. Give me the specific keywords (two words or less each) directly from the sample review without changing or editing in any way that helped you in categorization. Do not repeat keywords that have very similar meaning. 
            3. Provide these in a JSON format in the following way -
            
            {{ 'theme' : [Keyword 1, Keyword 2...], 
            'theme' : [Keyword 1, Keyword 2...],...}}
                        
                Review: {review_text}

                Return only the JSON object, nothing else."""
            }])

        # Get the response content
        response_text = message.content[0].text

        # Parse the JSON response
        return json.loads(response_text)
    except Exception as e:
        print(f"Error processing review: {e}")


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Initialize Anthropic client
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    if not anthropic_key:
        sys.exit('ANTHROPIC_API_KEY environment variable must be set')

    client = anthropic.Anthropic(api_key=anthropic_key)

    # Read input CSV file
    input_file = "attached_assets/Jewelry Store Google Map Reviews.csv"
    if not os.path.exists(input_file):
        sys.exit(f"Input file not found: {input_file}")

    # Read the CSV
    df = pd.read_csv(input_file)
    df = df[299:] # testing limit
    print(df.tail())
    df['raw_tags'] = 'Failed'
    df['themes'] = ''
    df['subthemes'] = ''

    # Check if 'Text' column exists
    if 'Text' not in df.columns:
        sys.exit("CSV file must contain a 'Text' column")

    # Process each review
    master_themes = {
        'Exceptional Customer Service & Support': [],
        'Poor Service & Long Wait Times': [],
        'Product Durability & Quality Issues': [],
        'Aesthetic Design & Visual Appeal': [],
        'Professional Piercing Services & Environment': [],
        'Piercing Complications & Jewelry Quality': [],
        'Store Ambiance & Try-On Experience': [],
        'Price & Policy Transparency': [],
        'Store Organization & Product Selection': [],
        'Complex Returns & Warranty Handling': [],
        'Communication & Policy Consistency': [],
        'Value & Price-Quality Assessment': [],
        'Affordable Luxury & Investment Value': [],
        'Online Shopping Experience': [],
        'Inventory & Cross-Channel Integration': []
    }
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for idx, row in df.iterrows():
        print(f"Processing review {idx + 1}/{len(df)}")
        for attempt in range(max_retries):
            try:

        THEMES_LIST = """ 
                    Exceptional Customer Service & Support
                    Poor Service & Long Wait Times
                    Product Durability & Quality Issues
                    Aesthetic Design & Visual Appeal
                    Professional Piercing Services & Environment
                    Piercing Complications & Jewelry Quality
                    Store Ambiance & Try-On Experience
                    Price & Policy Transparency
                    Store Organization & Product Selection
                    Complex Returns & Warranty Handling
                    Communication & Policy Consistency
                    Value & Price-Quality Assessment
                    Affordable Luxury & Investment Value
                    Online Shopping Experience
                    Inventory & Cross-Channel Integration """
        raw_themes = process_review_with_llm(client, THEMES_LIST, row['Text'])
        print(raw_themes)
        themes = ' | '.join(raw_themes.keys())
        subthemes = ' | '.join(
            ' | '.join(theme) if isinstance(theme, list) else theme
            for theme in raw_themes.values())

        # Add all JSON fields to the results
                df.iloc[idx, -3] = str(raw_themes)
                df.iloc[idx, -2] = themes
                df.iloc[idx, -1] = subthemes
                print(df.iloc[idx])
                
                # update theme master list
                for theme in raw_themes.keys():
                    if theme in master_themes:
                        master_themes[theme].append(raw_themes[theme])
                break  # Success - exit retry loop
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    print(f"All retries failed for review {idx + 1}: {str(e)}")
                    df.iloc[idx, -3] = f"Failed: {str(e)}"
                    df.iloc[idx, -2] = "Error"
                    df.iloc[idx, -1] = "Error"
        for theme in raw_themes.keys():
            if theme not in master_themes:
                pass
            else:
                master_themes[theme].append(raw_themes[theme])
        if (idx + 1) % 100 == 0:
            with open(f"master_themes_{timestamp}.json", "w") as outfile:
                json.dump(master_themes, outfile)

            output_file = f'reviews_tagged_{timestamp}.csv'

            # Save to CSV
            df.to_csv(output_file, index=False)
            print(
                f"Processed {idx + 1} reviews. Output saved to: {output_file}")

    # Generate output filename with timestamp
    for theme in master_themes.keys():
        vals = master_themes[theme]
        master_themes[theme] = list(set((itertools.chain.from_iterable(vals))))

    with open(f"master_themes_final_{timestamp}.json", "w") as outfile:
        json.dump(master_themes, outfile)

    output_file = f'reviews_tagged_{timestamp}.csv'

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Processing complete. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
