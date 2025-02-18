import json
import itertools
import pandas as pd
from ast import literal_eval  # Safer alternative to eval

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

# Load three JSON files as dictionaries
file_paths = [
    'attached_assets/reviews_tagged_final_1.json',
    'attached_assets/reviews_tagged_final_2.json',
    'attached_assets/reviews_tagged_final_3.json'
]

data1, data2, data3 = {}, {}, {}

with open(file_paths[0], 'r') as file1, \
     open(file_paths[1], 'r') as file2, \
     open(file_paths[2], 'r') as file3:

    data1 = json.load(file1)  # Already returns a dictionary if JSON is properly formatted
    data2 = json.load(file2)
    data3 = json.load(file3)  # Fixed typo (was loading from file2 twice before)


def process_theme_data(theme_dict):
    """Safely flatten and deduplicate theme values"""
    processed = {}
    for theme, values in theme_dict.items():
        # Ensure we're working with a list of lists
        if not all(isinstance(v, list) for v in values):
            print(f"Warning: Non-list values in {theme}")
            processed[theme] = list(set(values))
        else:
            flattened = list(set(itertools.chain.from_iterable(values)))
            processed[theme] = flattened
    return processed

data1 = process_theme_data(data1)
data2 = process_theme_data(data2)

# Combine data with missing key handling
for key in set().union(data1, data2, data3):
    combined = []
    combined.extend(data1.get(key, []))
    combined.extend(data2.get(key, []))
    combined.extend(data3.get(key, []))
    data3[key] = list(set(combined))  # Deduplicate final combined list

# Save with proper JSON dump
with open('attached_assets/reviews_tagged_final_0.json', 'w') as f:
    json.dump(data3, f, indent=2)  # ACTUAL WRITE OPERATION


# ========================
# 3. FIX CSV PROCESSING
# ========================

try:
    df = pd.read_csv('attached_assets/reviews_tagged_final_0.csv')
except FileNotFoundError:
    print("CSV file not found!")
    df = pd.DataFrame()

if not df.empty and 'raw_tags' in df.columns:
    for idx, row in df.iterrows():
        try:
            # Use safer literal_eval instead of eval
            raw_themes = literal_eval(row['raw_tags'])
            for theme, values in raw_themes.items():
                if theme in master_themes:
                    master_themes[theme].extend(values)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing row {idx}: {e}")
else:
    print("CSV data missing or invalid format")

# Final deduplication of master themes
for theme in master_themes:
    master_themes[theme] = list(set(master_themes[theme]))

# Save master themes
with open('attached_assets/reviews_tagged_final_0_csv.json', 'w') as f:
    json.dump(master_themes, f, indent=2)  # ACTUAL WRITE OPERATION

print("Processing complete. Check output files.")
