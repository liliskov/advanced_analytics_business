import json
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize the graph
G = nx.Graph()

# Read the JSON file line by line and process each JSON object
with open('/Users/lilivandermeersch/Desktop/AABD/Assignments/A4/memgraph-query-results-export6.json', 'r') as f:
    for line in f:
        try:
            record = json.loads(line)
            user1 = record['u']['properties']['screen_name']
            user2 = record['m']['properties']['screen_name']
            tweet_id = record['n']['properties']['ident']
            reply_count = record['n']['properties']['reply_count']
            retweet_count = record['n']['properties']['retweet_count']
            favorite_count = record['n']['properties']['favorite_count']

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

            # Add engagement metrics
            if 'tweets' not in G.nodes[user1]:
                G.nodes[user1]['tweets'] = {}
            G.nodes[user1]['tweets'][tweet_id] = {
                'reply_count': reply_count,
                'retweet_count': retweet_count,
                'favorite_count': favorite_count
            }

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

# Check if the graph has been populated
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    print("Graph is empty. Please check the input data and ensure it is correctly formatted.")
else:
    # Community Detection using Louvain method
    partition = community_louvain.best_partition(G, weight='weight')
    print("Community Detection Results:")
    print(partition)

    # Visualization
    plt.figure(figsize=(12, 8))

    # Draw the graph with community coloring
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap('viridis')
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40, cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('Twitter Network of Belgian Politicians')
    plt.show()

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    print("Degree Centrality:")
    print(degree_centrality)
    print("Betweenness Centrality:")
    print(betweenness_centrality)

    # Find the most influential users by degree centrality
    top_users_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Influential Users (Degree Centrality):")
    for user, centrality in top_users_degree:
        print(f"{user}: {centrality}")

    # Find the most influential users by betweenness centrality
    top_users_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Influential Users (Betweenness Centrality):")
    for user, centrality in top_users_betweenness:
        print(f"{user}: {centrality}")

    # Visualization of Influential Users by Degree Centrality
    users_degree = [user for user, _ in top_users_degree]
    centralities_degree = [centrality for _, centrality in top_users_degree]

    plt.figure(figsize=(10, 6))
    plt.barh(users_degree, centralities_degree, color='skyblue')
    plt.xlabel('Degree Centrality')
    plt.title('Influential Users by Degree Centrality')
    plt.gca().invert_yaxis()
    plt.show()

    # Visualization of Influential Users by Betweenness Centrality
    users_betweenness = [user for user, _ in top_users_betweenness]
    centralities_betweenness = [centrality for _, centrality in top_users_betweenness]

    plt.figure(figsize=(10, 6))
    plt.barh(users_betweenness, centralities_betweenness, color='skyblue')
    plt.xlabel('Betweenness Centrality')
    plt.title('Influential Users by Betweenness Centrality')
    plt.gca().invert_yaxis()
    plt.show()

    # Engagement Analysis
    engagement_data = defaultdict(lambda: {'replies': 0, 'retweets': 0, 'favorites': 0})
    for node in G.nodes(data=True):
        screen_name = node[0]
        if 'tweets' in node[1]:
            for tweet_id, metrics in node[1]['tweets'].items():
                engagement_data[screen_name]['replies'] += metrics['reply_count']
                engagement_data[screen_name]['retweets'] += metrics['retweet_count']
                engagement_data[screen_name]['favorites'] += metrics['favorite_count']

    # Convert engagement data to a list of tuples
    engagement_list = [(user, data['replies'], data['retweets'], data['favorites'])
                       for user, data in engagement_data.items()]

    # Sort by total engagement (replies + retweets + favorites)
    sorted_engagement = sorted(engagement_list, key=lambda x: x[1] + x[2] + x[3], reverse=True)[:10]

    print("Users by Engagement:")
    for user, replies, retweets, favorites in sorted_engagement:
        print(f"{user}: Replies={replies}, Retweets={retweets}, Favorites={favorites}")

    # Visualization of Engagement
    users_eng = [user for user, _, _, _ in sorted_engagement]
    replies = [replies for _, replies, _, _ in sorted_engagement]
    retweets = [retweets for _, _, retweets, _ in sorted_engagement]
    favorites = [favorites for _, _, _, favorites in sorted_engagement]

    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = range(len(users_eng))

    plt.bar(index, replies, bar_width, label='Replies')
    plt.bar([i + bar_width for i in index], retweets, bar_width, label='Retweets')
    plt.bar([i + bar_width * 2 for i in index], favorites, bar_width, label='Favorites')

    plt.xlabel('Users')
    plt.ylabel('Engagement Count')
    plt.title('Users by Engagement')
    plt.xticks([i + bar_width for i in index], users_eng, rotation=45)
    plt.legend()
    plt.show()

# Print top nodes by betweenness centrality
top_betweenness_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Nodes by Betweenness Centrality:")
for node, centrality in top_betweenness_nodes:
    print(f"Node: {node}, Betweenness Centrality: {centrality}")

# Visualization in Matplotlib (optional)
import matplotlib.pyplot as plt

nodes, centrality_values = zip(*top_betweenness_nodes)
plt.figure(figsize=(10, 6))
plt.barh(nodes, centrality_values, color='blue')
plt.xlabel('Betweenness Centrality')
plt.title('Nodes by Betweenness Centrality')
plt.gca().invert_yaxis()
plt.show()
