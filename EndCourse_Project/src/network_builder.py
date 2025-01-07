import networkx as nx
from typing import Set, Dict, Tuple
from collections import defaultdict

class NetworkBuilder:
    def __init__(self):
        self.G = nx.DiGraph()  # Đồ thị có hướng
        self.collab_counts = defaultdict(lambda: defaultdict(int))  # Đếm số lần collab

    def add_collaboration(self, channel_artist: str, collaborators: Set[str]):
        """Thêm các cạnh collab vào đồ thị"""
        # Thêm cạnh từ channel_artist đến mỗi collaborator
        for collaborator in collaborators:
            # Tăng số lần collab
            self.collab_counts[channel_artist][collaborator] += 1
            
            # Cập nhật/thêm cạnh với trọng số mới
            weight = self.collab_counts[channel_artist][collaborator]
            self.G.add_edge(channel_artist, collaborator, weight=weight)

    def save_network(self, output_path: str = 'data/network.gexf'):
        """Lưu đồ thị ra file"""
        # Lưu đồ thị dạng GEXF để giữ nguyên thuộc tính cạnh
        nx.write_gexf(self.G, output_path)
        print(f"\nĐã lưu mạng lưới collab vào {output_path}")
        
        # In thống kê cơ bản
        print(f"\nThống kê mạng lưới:")
        print(f"Số nghệ sĩ: {self.G.number_of_nodes()}")
        print(f"Số cạnh collab: {self.G.number_of_edges()}") 