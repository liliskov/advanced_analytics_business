import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the graph
G = nx.Graph()

# Read the JSON file line by line and process each JSON object
with open('/Users/lilivandermeersch/Desktop/AABD/Assignments/A4/memgraph-query-results-export.json', 'r') as f:
    for line in f:
        try:
            record = json.loads(line)
            user1 = record['u']['properties']['screen_name']
            user2 = record['m']['properties']['screen_name']
            tweet_text = record['n']['properties']['full_text']

            # Add nodes
            G.add_node(user1)
            G.add_node(user2)

            # Check if the edge already exists
            if G.has_edge(user1, user2):
                # Increment the weight if the edge already exists
                G[user1][user2]['weight'] += 1
            else:
                # Add the edge with initial weight
                G.add_edge(user1, user2, weight=1)

            # Add tweet text to nodes
            if 'tweets' not in G.nodes[user1]:
                G.nodes[user1]['tweets'] = []
            G.nodes[user1]['tweets'].append(tweet_text)

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

# Perform sentiment analysis
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    sentiment = analyzer.polarity_scores(text)
    return sentiment['compound']

# Collect sentiments for each politician
politicians = ['alexanderdecroo', 'tomvangrieken', 'SanderLoones', 'Bart_DeWever']
sentiment_data = {pol: {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0} for pol in politicians}

for pol in politicians:
    if pol in G.nodes:
        for tweet in G.nodes[pol].get('tweets', []):
            sentiment = get_sentiment(tweet)
            sentiment_data[pol]['total'] += 1
            if sentiment > 0.05:
                sentiment_data[pol]['positive'] += 1
            elif sentiment < -0.05:
                sentiment_data[pol]['negative'] += 1
            else:
                sentiment_data[pol]['neutral'] += 1

# Convert sentiment data to DataFrame for easier manipulation
sentiment_df = pd.DataFrame(sentiment_data).T
sentiment_df['positive_pct'] = sentiment_df['positive'] / sentiment_df['total']
sentiment_df['negative_pct'] = sentiment_df['negative'] / sentiment_df['total']
sentiment_df['neutral_pct'] = sentiment_df['neutral'] / sentiment_df['total']

# Plotting the sentiment proportions
sentiment_df[['positive_pct', 'negative_pct', 'neutral_pct']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xlabel('Politicians')
plt.ylabel('Proportion')
plt.title('Sentiment Analysis of Tweets Associated with Belgian Politicians')
plt.show()
