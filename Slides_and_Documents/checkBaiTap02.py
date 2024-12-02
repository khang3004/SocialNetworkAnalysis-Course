import networkx as nx
import numpy as np

# Bài tập 1: Mạng học tập (vô hướng)
G1 = nx.Graph()
nodes1 = ['An', 'Binh', 'Em', 'Dung', 'Cuong']
edges1 = [('An','Binh'), ('An','Em'), ('An','Dung'),
          ('Binh','Em'), ('Binh','Cuong'),
          ('Em','Dung'), ('Em','Cuong'),
          ('Dung','Cuong')]
G1.add_nodes_from(nodes1)
G1.add_edges_from(edges1)

# Bài tập 2: Mạng tổ chức (có hướng)
G2 = nx.DiGraph()
nodes2 = ['GD', 'P1', 'P2', 'P3', 'P4']
edges2 = [('P1','GD'), ('P2','GD'), ('P3','GD'), ('P4','GD'),
          ('P1','P2'), ('P2','P3'), ('P3','P4'), ('P4','P1')]
G2.add_nodes_from(nodes2)
G2.add_edges_from(edges2)

# Bài tập 3: Mạng xã hội (có hướng)
G3 = nx.DiGraph()
nodes3 = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6']
edges3 = [('U1','U2'), ('U1','U3'), ('U1','U4'),
          ('U2','U3'), ('U2','U4'), ('U2','U5'),
          ('U3','U4'), ('U3','U5'), ('U3','U6'),
          ('U4','U5'), ('U5','U6'), ('U6','U1')]
G3.add_nodes_from(nodes3)
G3.add_edges_from(edges3)

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
        print("\nDegree Centrality:")
        print({k: f"{v:.3f}" for k,v in deg_cent.items()})
    
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
    
    # 6. Clustering Coefficient (chỉ cho mạng vô hướng)
    if not G.is_directed():
        cluster = nx.clustering(G)
        print("\nClustering Coefficient:")
        print({k: f"{v:.3f}" for k,v in cluster.items()})

# # Phân tích cả 3 mạng
# analyze_network(G1, "Bài tập 1: Mạng học tập")
# analyze_network(G2, "Bài tập 2: Mạng tổ chức")
# analyze_network(G3, "Bài tập 3: Mạng xã hội trực tuyến")


def undirected_main():
    G = nx.Graph()
    nodes = ['An', 'Binh', 'Em', 'Dung', 'Cuong']
    edges = [('An','Binh'), ('An','Em'), ('An','Dung'),
              ('Binh','Em'), ('Binh','Cuong'),
          ('Em','Dung'), ('Em','Cuong'),
          ('Dung','Cuong')]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    analyze_network(G, "ĐỀ THI vô hướng")

def directed_main():
    DiG = nx.DiGraph()
    nodes = ['An', 'Binh', 'Em', 'Dung', 'Cuong']
    edges = [('An','Binh'), ('An','Em'), ('An','Dung'),
              ('Binh','Em'), ('Binh','Cuong'),
          ('Em','Dung'), ('Em','Cuong'),
          ('Dung','Cuong')]

    DiG.add_nodes_from(nodes)
    DiG.add_edges_from(edges)
    analyze_network(DiG, "Đồ thị có hướng")
if __name__ == "__main__":
    undirected_main()
    # directed_main()
    
