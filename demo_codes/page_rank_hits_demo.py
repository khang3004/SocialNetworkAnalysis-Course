import networkx as nx

# Tạo đồ thị ví dụ
G = nx.DiGraph()
G.add_edges_from([("A", "B"), ("B", "C"), ("C", "A"), ("A", "D"), ("D", "C")])

# Tính PageRank
pagerank = nx.pagerank(G, alpha=0.85)
print("PageRank:", pagerank)

# Tính HITS
hits = nx.hits(G)
print("Hubs:", hits[0])
print("Authorities:", hits[1])

# Chuyển đổi sang đồ thị vô hướng
G_undirected = G.to_undirected()

# Preferential Attachment với đồ thị vô hướng - sử dụng tên node thay vì chỉ số
preferential_attachment = list(nx.preferential_attachment(G_undirected, [("A", "D")]))
print("Preferential Attachment:", preferential_attachment)

# Resource Allocation Index với đồ thị vô hướng - sử dụng tên node thay vì chỉ số
resource_allocation = list(nx.resource_allocation_index(G_undirected, [("A", "D")]))
print("Resource Allocation Index:", resource_allocation)

