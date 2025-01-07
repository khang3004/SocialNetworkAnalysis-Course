import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import community  # python-louvain package
import json
import os

class NetworkAnalyzer:
    def __init__(self, network_path: str = 'data/network.gexf'):
        """Khởi tạo NetworkAnalyzer"""
        # Load network
        self.G = nx.read_gexf(network_path)
        
        # Tạo thư mục output nếu chưa tồn tại
        self.output_dirs = {
            'metrics': 'output/metrics',
            'communities': 'output/communities',
            'link_prediction': 'output/link_prediction',
            'visualizations': 'output/visualizations'
        }
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def analyze_basic_metrics(self) -> Dict:
        """Phân tích các metrics cơ bản của mạng"""
        metrics = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'avg_clustering': nx.average_clustering(self.G),
        }
        
        # Tìm thành phần liên thông lớn nhất
        if self.G.is_directed():
            largest_cc = max(nx.strongly_connected_components(self.G), key=len)
            largest_subgraph = self.G.subgraph(largest_cc)
            metrics['largest_strongly_connected_component_size'] = len(largest_cc)
            
            # Tính path length cho thành phần liên thông lớn nhất
            if len(largest_cc) > 1:
                metrics['avg_shortest_path_in_largest_component'] = nx.average_shortest_path_length(largest_subgraph)
                metrics['diameter_in_largest_component'] = nx.diameter(largest_subgraph)
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            largest_subgraph = self.G.subgraph(largest_cc)
            metrics['largest_connected_component_size'] = len(largest_cc)
            
            if len(largest_cc) > 1:
                metrics['avg_shortest_path_in_largest_component'] = nx.average_shortest_path_length(largest_subgraph)
                metrics['diameter_in_largest_component'] = nx.diameter(largest_subgraph)
        
        # Tính các metrics khác
        metrics.update({
            'avg_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'number_of_components': nx.number_strongly_connected_components(self.G) if self.G.is_directed() 
                                  else nx.number_connected_components(self.G),
            'reciprocity': nx.reciprocity(self.G) if self.G.is_directed() else 'undirected'
        })
        
        # Tính phân phối bậc
        in_degrees = [d for n, d in self.G.in_degree()] if self.G.is_directed() else None
        out_degrees = [d for n, d in self.G.out_degree()] if self.G.is_directed() else None
        degrees = [d for n, d in self.G.degree()]
        
        metrics['degree_stats'] = {
            'max_degree': max(degrees),
            'min_degree': min(degrees),
            'median_degree': np.median(degrees),
            'mean_degree': np.mean(degrees),
            'std_degree': np.std(degrees)
        }
        
        if self.G.is_directed():
            metrics['degree_stats'].update({
                'max_in_degree': max(in_degrees),
                'min_in_degree': min(in_degrees),
                'median_in_degree': np.median(in_degrees),
                'mean_in_degree': np.mean(in_degrees),
                'std_in_degree': np.std(in_degrees),
                'max_out_degree': max(out_degrees),
                'min_out_degree': min(out_degrees),
                'median_out_degree': np.median(out_degrees),
                'mean_out_degree': np.mean(out_degrees),
                'std_out_degree': np.std(out_degrees)
            })
        
        # Vẽ phân phối bậc
        plt.figure(figsize=(10, 6))
        plt.hist(degrees, bins=30, alpha=0.7, label='Degree')
        if self.G.is_directed():
            plt.hist(in_degrees, bins=30, alpha=0.5, label='In-Degree')
            plt.hist(out_degrees, bins=30, alpha=0.5, label='Out-Degree')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title('Degree Distribution')
        plt.legend()
        plt.savefig(f"{self.output_dirs['visualizations']}/degree_distribution.png")
        plt.close()
        
        # Lưu metrics
        with open(f"{self.output_dirs['metrics']}/basic_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
            
        return metrics

    def analyze_centrality_metrics(self) -> Dict[str, Dict]:
        """Phân tích các centrality metrics và lưu top-10"""
        centrality_metrics = {
            'degree': nx.degree_centrality(self.G),
            'betweenness': nx.betweenness_centrality(self.G),
            'closeness': nx.closeness_centrality(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G),
            'pagerank': nx.pagerank(self.G)
        }
        
        # Tạo DataFrame cho mỗi metric và lưu top-10
        for metric_name, metric_values in centrality_metrics.items():
            df = pd.DataFrame.from_dict(metric_values, orient='index', columns=[metric_name])
            df.sort_values(by=metric_name, ascending=False, inplace=True)
            
            # Lưu top-10
            top10_df = df.head(10)
            top10_df.to_csv(f"{self.output_dirs['metrics']}/top10_{metric_name}.csv")
            
            # Visualize top-10
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top10_df.reset_index(), x='index', y=metric_name)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Top 10 Artists by {metric_name.capitalize()} Centrality')
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['visualizations']}/top10_{metric_name}.png")
            plt.close()
            
        return centrality_metrics

    def detect_communities(self) -> Dict[str, Dict]:
        """Phát hiện cộng đồng bằng nhiều thuật toán"""
        community_results = {}
        
        # Chuyển đổi đồ thị có hướng thành vô hướng cho các thuật toán
        G_undirected = self.G.to_undirected()
        
        # 1. Louvain Method (Modularity Optimization)
        print("\nDetecting communities using Louvain method...")
        louvain_communities = community.best_partition(G_undirected)
        
        # 2. Label Propagation (trên đồ thị vô hướng)
        print("Detecting communities using Label Propagation...")
        lpa_communities = list(nx.community.label_propagation_communities(G_undirected))
        lpa_dict = {node: i for i, comm in enumerate(lpa_communities) for node in comm}
        
        # 3. Girvan-Newman
        print("Detecting communities using Girvan-Newman...")
        gn_communities = list(nx.community.girvan_newman(G_undirected))
        gn_dict = {node: i for i, comm in enumerate(tuple(gn_communities[0])) for node in comm}
        
        # 4. K-means clustering
        print("Detecting communities using K-means clustering...")
        features = self._prepare_clustering_features()
        n_clusters = min(5, len(self.G.nodes()))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(features)
        
        # 5. Gaussian Mixture
        print("Detecting communities using Gaussian Mixture...")
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(features)
        
        # 6. DBSCAN
        print("Detecting communities using DBSCAN...")
        # Tính toán eps dựa trên dữ liệu
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        distances = np.sort(np.sqrt(np.sum((scaled_features[:, None, :] - 
                                          scaled_features[None, :, :]) ** 2, axis=2)), axis=1)
        eps = np.mean(distances[:, 1])  # Lấy khoảng cách trung bình đến điểm gần nhất
        
        dbscan = DBSCAN(eps=eps, min_samples=min(5, len(self.G.nodes())-1))
        dbscan_labels = dbscan.fit_predict(scaled_features)
        
        # Lưu kết quả và đánh giá
        algorithms = {
            'louvain': louvain_communities,
            'lpa': lpa_dict,
            'girvan_newman': gn_dict,
            'kmeans': dict(enumerate(kmeans_labels)),
            'gmm': dict(enumerate(gmm_labels)),
            'dbscan': dict(enumerate(dbscan_labels))
        }
        
        # Đánh giá và so sánh
        print("\nEvaluating community detection results...")
        comparison_metrics = self._evaluate_communities(algorithms)
        
        # Lưu kết quả chi tiết của từng thuật toán
        for algo_name, communities in algorithms.items():
            print(f"\nSaving and visualizing results for {algo_name}...")
            # Tạo danh sách các cộng đồng
            community_lists = {}
            for node, comm_id in communities.items():
                # Chuyển đổi comm_id thành int thông thường
                comm_id = int(comm_id) if isinstance(comm_id, (np.int32, np.int64)) else comm_id
                if comm_id not in community_lists:
                    community_lists[comm_id] = []
                community_lists[comm_id].append(node)
            
            # Lưu kết quả
            with open(f"{self.output_dirs['communities']}/{algo_name}_communities.json", 'w', encoding='utf-8') as f:
                json.dump(community_lists, f, indent=4, ensure_ascii=False)
            
            # Visualize từng phương pháp
            self._visualize_communities(communities, algo_name)
        
        # Lưu bảng so sánh
        comparison_df = pd.DataFrame(comparison_metrics).round(4)
        comparison_df.to_csv(f"{self.output_dirs['communities']}/comparison_table.csv")
        
        # In kết quả tóm tắt
        print("\nCommunity Detection Summary:")
        for algo_name, metrics in comparison_metrics.items():
            print(f"\n{algo_name}:")
            print(f"- Number of communities: {metrics['number_of_communities']}")
            print(f"- Modularity: {metrics['modularity']:.4f}")
            print(f"- Average community size: {metrics['avg_community_size']:.2f}")
        
        return comparison_metrics

    def predict_links(self) -> Dict[str, float]:
        """Dự đoán liên kết bằng nhiều phương pháp"""
        # Chia tập dữ liệu
        edges = list(self.G.edges())
        np.random.shuffle(edges)
        train_size = int(0.8 * len(edges))
        
        train_edges = edges[:train_size]
        test_edges = edges[train_size:]
        
        # Tạo đồ thị training và chuyển thành dạng vô hướng
        G_train = self.G.copy()
        G_train.remove_edges_from(test_edges)
        G_train_undirected = G_train.to_undirected()
        
        # Các phương pháp dự đoán truyền thống
        predictors = {
            'jaccard': nx.jaccard_coefficient(G_train_undirected),
            'adamic_adar': nx.adamic_adar_index(G_train_undirected),
            'preferential_attachment': nx.preferential_attachment(G_train_undirected),
            'resource_allocation': nx.resource_allocation_index(G_train_undirected)
        }
        
        # Đánh giá các phương pháp
        results = self._evaluate_link_prediction(predictors, test_edges)
        
        # Lưu kết quả
        with open(f"{self.output_dirs['link_prediction']}/prediction_results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

    def _prepare_clustering_features(self) -> np.ndarray:
        """Chuẩn bị features cho clustering"""
        features = []
        for node in self.G.nodes():
            node_features = [
                self.G.degree(node),
                nx.clustering(self.G, node),
                nx.closeness_centrality(self.G)[node],
                nx.betweenness_centrality(self.G)[node]
            ]
            features.append(node_features)
            
        # Chuẩn hóa features
        scaler = StandardScaler()
        return scaler.fit_transform(features)

    def _evaluate_communities(self, algorithms: Dict) -> Dict:
        """Đánh giá kết quả phát hiện cộng đồng"""
        metrics = {}
        G_undirected = self.G.to_undirected()
        
        for algo_name, communities in algorithms.items():
            # Chuyển đổi communities thành dạng list of sets
            community_sets = {}
            for node, comm_id in communities.items():
                if comm_id not in community_sets:
                    community_sets[comm_id] = set()
                community_sets[comm_id].add(node)
            
            # Tính các metrics
            n_communities = len(set(communities.values()))
            avg_size = len(self.G.nodes()) / n_communities if n_communities > 0 else 0
            
            # Tính modularity
            try:
                modularity = nx.community.modularity(G_undirected, community_sets.values())
            except:
                # Nếu không tính được modularity, gán giá trị mặc định
                modularity = 0.0
            
            # Tính silhouette score cho các thuật toán clustering
            if algo_name in ['KMeans', 'GaussianMixture', 'DBSCAN']:
                features = self._prepare_clustering_features()
                try:
                    silhouette = silhouette_score(features, list(communities.values()))
                except:
                    silhouette = 0.0
            else:
                silhouette = 0.0
            
            metrics[algo_name] = {
                'number_of_communities': n_communities,
                'avg_community_size': avg_size,
                'modularity': modularity,
                'silhouette_score': silhouette
            }
        
        return metrics

    def _evaluate_link_prediction(self, predictors: Dict, test_edges: List) -> Dict:
        """Đánh giá kết quả dự đoán liên kết"""
        results = {}
        
        for name, predictor in predictors.items():
            # Thu thập các dự đoán
            predictions = []
            scores = []
            for u, v, score in predictor:
                predictions.append((u, v))
                scores.append(score)
            
            # Tính các metrics
            true_positives = sum(1 for edge in test_edges if edge in predictions or edge[::-1] in predictions)
            precision = true_positives / len(predictions) if predictions else 0
            recall = true_positives / len(test_edges) if test_edges else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'number_of_predictions': len(predictions)
            }
        
        return results

    def visualize_network(self) -> None:
        """Trực quan hóa mạng lưới với các metrics"""
        plt.figure(figsize=(20, 20))
        
        # Sử dụng layout phù hợp
        pos = nx.spring_layout(self.G, k=1, iterations=50)
        
        # Vẽ nodes với size dựa trên degree và color dựa trên betweenness
        node_sizes = [self.G.degree(node) * 100 for node in self.G.nodes()]
        bc = nx.betweenness_centrality(self.G)
        
        nodes = nx.draw_networkx_nodes(
            self.G, pos,
            node_size=node_sizes,
            node_color=list(bc.values()),
            cmap=plt.cm.viridis
        )
        
        # Vẽ edges với độ trong suốt
        nx.draw_networkx_edges(self.G, pos, alpha=0.2)
        
        # Thêm labels cho nodes có degree cao
        labels = {}
        mean_degree = np.mean([self.G.degree(n) for n in self.G.nodes()])
        for node in self.G.nodes():
            if self.G.degree(node) > mean_degree:
                labels[node] = node
                
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
        
        plt.title("VPop Artists Collaboration Network", fontsize=16, pad=20)
        plt.colorbar(nodes, label='Betweenness Centrality')
        plt.axis('off')
        
        # Lưu visualization
        plt.savefig(f"{self.output_dirs['visualizations']}/network_visualization.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_communities(self, communities: Dict, algo_name: str) -> None:
        """Trực quan hóa kết quả phát hiện cộng đồng"""
        plt.figure(figsize=(15, 15))
        
        # Sử dụng layout phù hợp
        pos = nx.spring_layout(self.G)
        
        # Vẽ nodes với màu sắc theo cộng đồng
        unique_communities = set(communities.values())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_communities)))
        color_map = dict(zip(unique_communities, colors))
        
        # Đảm bảo tất cả nodes đều có community
        node_colors = []
        for node in self.G.nodes():
            try:
                node_colors.append(color_map[communities[node]])
            except KeyError:
                # Nếu node không có trong communities, gán màu mặc định
                node_colors.append([0.7, 0.7, 0.7, 1.0])  # Màu xám
                print(f"Warning: Node {node} not found in communities for {algo_name}")
        
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=100)
        nx.draw_networkx_edges(self.G, pos, alpha=0.2)
        
        # Thêm labels cho nodes có degree cao
        labels = {}
        mean_degree = np.mean([self.G.degree(n) for n in self.G.nodes()])
        for node in self.G.nodes():
            if self.G.degree(node) > mean_degree:
                labels[node] = node
        nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
        
        plt.title(f"Community Detection using {algo_name}", fontsize=16, pad=20)
        plt.axis('off')
        
        # Lưu visualization
        plt.savefig(f"{self.output_dirs['visualizations']}/communities_{algo_name}.png",
                    dpi=300, bbox_inches='tight')
        plt.close() 