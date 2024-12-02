import networkx as nx
import matplotlib.pyplot as plt

# Tạo đồ thị mạng xã hội
G = nx.Graph()

# Thêm các nút
G.add_nodes_from(["Alice", "Bob", "Charlie", "David", "Eve"])

# Thêm các cạnh
G.add_edges_from([("Alice", "Bob"), ("Alice", "David"), ("Bob", "Charlie"), ("Charlie", "David"), ("David", "Eve")])

G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5)])

# Common Neighbors
common_neighbors = list(nx.common_neighbors(G, 1, 4))
print("Common Neighbors:", common_neighbors)

# Jaccard Coefficient
jaccard = list(nx.jaccard_coefficient(G, [(1, 4)]))
print("Jaccard Coefficient:", jaccard)


# Adamic-Adar Index
adamic_adar = list(nx.adamic_adar_index(G, [(1, 4)]))
print("Adamic-Adar Index:", adamic_adar)

# Vẽ đồ thị
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='skyblue', font_weight='bold', node_size=1500)
plt.title("Social Network Graph")
plt.show()

# Tính độ trung tâm trong đồ thị mạng xã hội
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)

import community.community_louvain as community_louvain
import matplotlib.cm as cm

# Sử dụng Louvain Method để tìm cộng đồng
partition = community_louvain.best_partition(G)

# Vẽ cộng đồng với màu sắc khác nhau
pos = nx.spring_layout(G)
cmap = plt.colormaps['viridis']
plt.figure(figsize=(8, 6))

for node, community in partition.items():
    nx.draw_networkx_nodes(G, pos, [node], node_size=500, node_color=[cmap(community)])
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.title("Communities in Social Network")
plt.show()
