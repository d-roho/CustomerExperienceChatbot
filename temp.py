def fetch_sentiment(self, client):
  """Fetch sentiment for given filters and concepts."""
  try:
      counter = 0
      themes = ['Aesthetic Design & Visual Appeal', 'Online Shopping Experience']

      for theme in themes:
          concept = {
              "type":
              "concept_list",
              'name':
              theme}

          result = client.get('/concepts/sentiment/', concept_selector=concept, filter=filter
                          )

          rows = []
          for concept in result['match_counts']:
              row = {
                  'Theme Name': concept['name'],
                  'Proportion of Subset With Theme': concept['match_count'],
                  'Proportion of Positive Mentions': concept['sentiment_share']['positive'],
                  'Proportion of Neutral Mentions': concept['sentiment_share']['neutral'],
                  'Proportion of Negative Mentions': concept['sentiment_share']['negative']
              }
              rows.append(row)
              filter_count = result['filter_count'] 

          # Create DataFrame
          df = pd.DataFrame(rows)
          df['Proportion of Subset With Theme'] = df['Proportion of Subset With Theme']/filter_count 
          if counter == 0:
              df_deep_copy = df.copy(deep=True)
          else:
              df_to_merge = df.copy(deep=True)
              df_deep_copy = pd.concat([df_deep_copy, df_to_merge])
          counter +=1
          print(len(df_deep_copy))


      return df_deep_copy

  except Exception as e:
      raise RuntimeError(f"Failed to Fetch Drivers: {str(e)}")



