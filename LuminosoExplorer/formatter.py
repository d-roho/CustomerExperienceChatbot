import json

with open('reviews_tagged_final_0.json', 'r') as file:
  unformatted_tags = json.load(file)  # Already returns a dictionary if JSON is properly formatted

colors = [
  "#808080", "#800080", "#800000",
  "#008080", "#000080", "#808000",
  "#FF0000", "#00FF00", "#0000FF",
  "#808080", "#800080", "#800000",
  "#008080", "#000080", "#808000",
  "#FF0000", "#00FF00", "#0000FF"
]

output = []
color_picker = 0
for concept_list_name, phrases in unformatted_tags.items():
  color_idx = color_picker % len(colors)
  color_picker += 1
  concepts = []
  for idx, phrase in enumerate(phrases):
      color = colors[idx % len(colors)]
      concepts.append({
          "texts": phrase,
          "name": phrase,
          "color": colors[color_idx]
      })
  output.append({
      "concept_list_name": concept_list_name,
      "concepts": concepts
  })

# Write formatted JSON to file
with open('themes_formatted.json', 'w') as f:
    json.dump(output, f, indent=2)  # Directly dump to file
