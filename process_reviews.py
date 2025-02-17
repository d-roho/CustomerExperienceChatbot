import pandas as pd
import anthropic
import json
from datetime import datetime
import os
import sys
from pathlib import Path

def process_review_with_llm(client, review_text):
    """Process a single review through Claude and get structured output"""
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # newest Anthropic model as of Oct 22, 2024
            max_tokens=1000,
            temperature=0,
            system="You are a helpful assistant that analyzes customer reviews. Output only valid JSON.",
            messages=[{
                "role": "user",
                "content": f"""Analyze this review and output JSON with the following fields:
                - sentiment: (positive, negative, or neutral)
                - key_topics: array of main topics mentioned
                - rating_estimate: estimated rating out of 5 based on sentiment
                - main_feedback: primary feedback point
                
                Review: {review_text}

                Return only the JSON object, nothing else."""
            }]
        )
        
        # Get the response content
        response_text = message.content[0].text
        
        # Parse the JSON response
        return json.loads(response_text)
    except Exception as e:
        print(f"Error processing review: {e}")
        return {
            "sentiment": "error",
            "key_topics": [],
            "rating_estimate": 0,
            "main_feedback": f"Error processing: {str(e)}"
        }

def main():
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
    
    # Check if 'Text' column exists
    if 'Text' not in df.columns:
        sys.exit("CSV file must contain a 'Text' column")
    
    # Process each review
    results = []
    for idx, row in df.iterrows():
        print(f"Processing review {idx + 1}/{len(df)}")
        analysis = process_review_with_llm(client, row['Text'])
        
        # Add all JSON fields to the results
        result_row = row.to_dict()
        for key, value in analysis.items():
            result_row[f'llm_{key}'] = value
        results.append(result_row)
    
    # Create new dataframe with results
    result_df = pd.DataFrame(results)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'Reviews_tagged_{timestamp}.csv'
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Processing complete. Output saved to: {output_file}")

if __name__ == "__main__":
    main()
