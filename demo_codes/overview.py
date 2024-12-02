import pandas as pd
import networkx as nx

# Load dữ liệu và xây dựng đồ thị
edges = pd.read_csv("twitter_edges.csv")  # giả sử có tập dữ liệu twitter
G = nx.from_pandas_edgelist(edges, source='source', target='target')

# Tính toán các chỉ số
pagerank = nx.pagerank(G)
closeness = nx.closeness_centrality(G)
betweenness = nx.betweenness_centrality(G)

# Phát hiện cộng đồng
from community import community_louvain
partition = community_louvain.best_partition(G)

print("PageRank:", pagerank)
print("Closeness Centrality:", closeness)
print("Betweenness Centrality:", betweenness)
print("Community Partition:", partition)
