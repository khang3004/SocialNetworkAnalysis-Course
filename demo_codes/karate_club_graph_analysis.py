import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# G = nx.karate_club_graph()

# print(G.nodes())
# print(G.edges())    

# plt.figure(figsize=(12, 8))
# nx.draw(G, with_labels=True)
# plt.show()

# # Thông tin đồ thị
# print("Thông tin đồ thị:")
# print(f"Số đỉnh: {G.number_of_nodes()}")
# print(f"Số cạnh: {G.number_of_edges()}")
# print(f"Bậc trung bình: {sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")



def analyze_network(G, name=""):
    print(f"\n=== Phân tích {name} ===")
    
    # 1. Mật độ mạng
    density = nx.density(G)
    print(f"Mật độ mạng: {density:.3f}")
    
    # 2. Bậc của các đỉnh
    if G.is_directed():
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        print("\nBậc vào và ra:")
        for node in G.nodes():
            print(f"{node}: in={in_deg[node]}, out={out_deg[node]}")
    else:
        degrees = dict(G.degree())
        print("\nBậc của các đỉnh:")
        for node, deg in degrees.items():
            print(f"{node}: {deg}")
    
    # 3. Degree Centrality
    if G.is_directed():
        in_cent = nx.in_degree_centrality(G)
        out_cent = nx.out_degree_centrality(G)
        print("\nDegree Centrality:")
        print("In-degree centrality:", {k: f"{v:.3f}" for k,v in in_cent.items()})
        print("Out-degree centrality:", {k: f"{v:.3f}" for k,v in out_cent.items()})
    else:
        deg_cent = nx.degree_centrality(G)
        deg_cent_df = pd.DataFrame(list(deg_cent.items()), columns=["Node", "Degree Centrality"]).sort_values(by="Degree Centrality", ascending=False)
        print("\nDegree Centrality:")
        print({k: f"{v:.3f}" for k,v in deg_cent.items()})
        print(deg_cent_df.head())
        title = f"{name} - Degree Centrality"
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(deg_cent_df["Node"], deg_cent_df["Degree Centrality"])
        plt.show()
    
    # 4. Closeness Centrality
    if G.is_directed():
        in_close = nx.closeness_centrality(G.reverse())
        out_close = nx.closeness_centrality(G)
        print("\nCloseness Centrality:")
        print("In-closeness:", {k: f"{v:.3f}" for k,v in in_close.items()})
        print("Out-closeness:", {k: f"{v:.3f}" for k,v in out_close.items()})
    else:
        close = nx.closeness_centrality(G)
        print("\nCloseness Centrality:")
        print({k: f"{v:.3f}" for k,v in close.items()})
    
    # 5. Betweenness Centrality
    between = nx.betweenness_centrality(G)
    print("\nBetweenness Centrality:")
    print({k: f"{v:.3f}" for k,v in between.items()})
    between_df = pd.DataFrame(list(between.items()), columns=["Node", "Betweenness Centrality"]).sort_values(by="Betweenness Centrality", ascending=False)
    print(between_df.head())
    title = f"{name} - Betweenness Centrality"
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.bar(between_df["Node"], between_df["Betweenness Centrality"])
    plt.show()
    # 6. Clustering Coefficient (chỉ cho mạng vô hướng)
    if not G.is_directed():
        cluster = nx.clustering(G)
        print("\nClustering Coefficient:")
        print({k: f"{v:.3f}" for k,v in cluster.items()})

if __name__ == "__main__":
    G = nx.karate_club_graph()
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", font_size=10, font_weight="bold", )
    plt.show()
    analyze_network(G, "Karate Club")


