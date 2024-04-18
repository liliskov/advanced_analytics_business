import pandas as pd
import json

with open('dataset.json') as file:
    data = json.load(file)

df = pd.json_normalize(data, meta=['appid', 'release', 'title', 'price', 'sentiment',
                            'reviews', 'percentage', 'tags', 'screenshots'])
df.columns = ['appid', 'release', 'title', 'price', 'sentiment',
                            'reviews', 'percentage', 'tags', 'screenshots']
df.to_csv('dataset.csv')