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
        
        # Phân tích phân phối bậc và kiểm tra tính scale-free
        def fit_power_law(data):
            """Fit phân phối power law và tính các tham số"""
            data = np.array(data)
            # Chỉ lấy các giá trị lớn hơn 0 để tính toán
            data = data[data > 0]
            log_data = np.log(data)
            
            # Tính alpha (hệ số mũ) bằng MLE
            n = len(data)
            alpha = 1 + n / np.sum(log_data - np.log(np.min(data)))
            
            # Tính Kolmogorov-Smirnov statistic
            from scipy import stats
            theoretical_cdf = lambda x: 1 - (x/np.min(data))**(1-alpha)
            D, p_value = stats.kstest(data, theoretical_cdf)
            
            return alpha, D, p_value

        # Phân tích phân phối bậc
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Histogram thông thường
        plt.subplot(2, 2, 1)
        plt.hist(degrees, bins='auto', alpha=0.7, label='Degree', density=True)
        if self.G.is_directed():
            plt.hist(in_degrees, bins='auto', alpha=0.5, label='In-Degree', density=True)
            plt.hist(out_degrees, bins='auto', alpha=0.5, label='Out-Degree', density=True)
        plt.xlabel('Degree (k)')
        plt.ylabel('P(k)')
        plt.title('Degree Distribution')
        plt.legend()
        
        # Subplot 2: Log-log plot với power law fit
        plt.subplot(2, 2, 2)
        
        # Tính phân phối thực tế (loại bỏ các giá trị 0)
        degrees_nonzero = np.array(degrees)[np.array(degrees) > 0]
        unique_degrees, counts = np.unique(degrees_nonzero, return_counts=True)
        prob = counts / len(degrees_nonzero)
        
        # Plot log-log
        plt.loglog(unique_degrees, prob, 'bo', label='Observed', alpha=0.6)
        
        # Fit và plot power law
        alpha, D, p_value = fit_power_law(degrees_nonzero)
        x_min = min(degrees_nonzero)
        x_max = max(degrees_nonzero)
        x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_fit = x_fit**(-alpha)
        # Normalize để khớp với dữ liệu thực tế
        y_fit = y_fit * (prob[0] / y_fit[0])
        plt.loglog(x_fit, y_fit, 'r-', label=f'Power Law (α={alpha:.2f})')
        
        plt.xlabel('Degree (k)')
        plt.ylabel('P(k)')
        plt.title('Log-Log Degree Distribution')
        plt.legend()
        
        # Subplot 3: CCDF (Complementary Cumulative Distribution Function)
        plt.subplot(2, 2, 3)
        sorted_degrees = np.sort(degrees_nonzero)
        ccdf = 1 - np.arange(len(sorted_degrees)) / float(len(sorted_degrees))
        
        plt.loglog(sorted_degrees, ccdf, 'bo', label='Observed CCDF', alpha=0.6)
        
        # Theoretical CCDF for power law (chỉ tính cho các giá trị dương)
        ccdf_theory = (x_fit/x_min)**(1-alpha)
        plt.loglog(x_fit, ccdf_theory * ccdf[0] / ccdf_theory[0], 'r-', 
                  label='Power Law CCDF')
        
        plt.xlabel('Degree (k)')
        plt.ylabel('P(K ≥ k)')
        plt.title('CCDF of Degree Distribution')
        plt.legend()
        
        # Thêm thông tin kiểm định vào subplot 4
        plt.subplot(2, 2, 4)
        plt.axis('off')
        info_text = (
            f"Power Law Analysis Results:\n\n"
            f"Exponent (α) = {alpha:.3f}\n"
            f"KS Statistic (D) = {D:.3f}\n"
            f"p-value = {p_value:.3f}\n\n"
            f"Scale-free Test:\n"
            f"{'Network appears scale-free' if p_value > 0.05 else 'Network may not be scale-free'}\n"
            f"(p > 0.05 suggests scale-free property)\n\n"
            f"Additional Metrics:\n"
            f"Mean degree: {np.mean(degrees):.2f}\n"
            f"Median degree: {np.median(degrees):.2f}\n"
            f"Max degree: {max(degrees)}\n"
            f"Min degree: {min(degrees)}"
        )
        plt.text(0.1, 0.9, info_text, fontsize=10, va='top')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dirs['visualizations']}/degree_distribution_analysis.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Thêm kết quả phân tích scale-free vào metrics
        metrics['scale_free_analysis'] = {
            'power_law_exponent': float(alpha),
            'ks_statistic': float(D),
            'p_value': float(p_value),
            'is_scale_free': bool(p_value > 0.05)
        }
        
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

        # Tính toán các chỉ số tương tự cho tất cả các cặp nodes
        similarity_scores = {}
        nodes = list(G_train.nodes())
        n_nodes = len(nodes)
        
        # 1. Jaccard Coefficient
        jaccard_dict = {}
        for u, v, score in nx.jaccard_coefficient(G_train_undirected):
            jaccard_dict[(u, v)] = score
        similarity_scores['jaccard'] = jaccard_dict
        
        # 2. Adamic-Adar Index
        adamic_adar_dict = {}
        for u, v, score in nx.adamic_adar_index(G_train_undirected):
            adamic_adar_dict[(u, v)] = score
        similarity_scores['adamic_adar'] = adamic_adar_dict
        
        # 3. Preferential Attachment
        pa_dict = {}
        for u, v, score in nx.preferential_attachment(G_train_undirected):
            pa_dict[(u, v)] = score
        similarity_scores['preferential_attachment'] = pa_dict
        
        # 4. Resource Allocation Index
        ra_dict = {}
        for u, v, score in nx.resource_allocation_index(G_train_undirected):
            ra_dict[(u, v)] = score
        similarity_scores['resource_allocation'] = ra_dict
        
        # 5. Common Neighbors
        cn_dict = {}
        for u in nodes:
            for v in nodes:
                if u != v:
                    cn_dict[(u, v)] = len(list(nx.common_neighbors(G_train_undirected, u, v)))
        similarity_scores['common_neighbors'] = cn_dict
        
        # 6. Cosine Similarity
        cosine_dict = {}
        adj_matrix = nx.adjacency_matrix(G_train_undirected).toarray()  # Chuyển sang numpy array
        for i, u in enumerate(nodes):
            u_vector = adj_matrix[i, :]  # Lấy vector hàng
            for j, v in enumerate(nodes):
                if i < j:  # Chỉ tính cho các cặp không trùng lặp
                    v_vector = adj_matrix[j, :]  # Lấy vector hàng
                    numerator = np.dot(u_vector, v_vector)  # Tích vô hướng
                    denominator = np.sqrt(np.dot(u_vector, u_vector) * 
                                        np.dot(v_vector, v_vector))
                    if denominator != 0:
                        cosine_dict[(u, v)] = float(numerator / denominator)  # Chuyển về float
                    else:
                        cosine_dict[(u, v)] = 0.0
        similarity_scores['cosine'] = cosine_dict
        
        # 7. SimRank (simplified version for performance)
        simrank_dict = {}
        C = 0.8  # decay factor
        max_iter = 5
        
        # Initialize SimRank scores
        sim_old = {(u,v): 1.0 if u == v else 0.0 for u in nodes for v in nodes}
        
        # Iterate to compute SimRank
        for _ in range(max_iter):
            sim_new = {}
            for u in nodes:
                for v in nodes:
                    if u == v:
                        sim_new[(u,v)] = 1.0
                    else:
                        u_neighbors = list(G_train.predecessors(u))
                        v_neighbors = list(G_train.predecessors(v))
                        if not u_neighbors or not v_neighbors:
                            sim_new[(u,v)] = 0.0
                        else:
                            sum_sim = 0.0
                            for u_nb in u_neighbors:
                                for v_nb in v_neighbors:
                                    sum_sim += sim_old[(u_nb,v_nb)]
                            sim_new[(u,v)] = (C * sum_sim / 
                                          (len(u_neighbors) * len(v_neighbors)))
            sim_old = sim_new
        
        similarity_scores['simrank'] = {(u,v): score for (u,v), score in sim_old.items() 
                                      if u != v}
        
        # Tạo DataFrames cho từng phương pháp và lưu kết quả
        for method, scores in similarity_scores.items():
            # Convert to DataFrame
            df = pd.DataFrame([(u, v, score) for (u,v), score in scores.items()],
                             columns=['Node1', 'Node2', 'Similarity'])
            
            # Sắp xếp theo điểm số giảm dần
            df = df.sort_values('Similarity', ascending=False)
            
            # Lưu top pairs
            top_pairs = df.head(20)
            top_pairs.to_csv(f"{self.output_dirs['link_prediction']}/top_pairs_{method}.csv",
                            index=False)
            
            # Visualize top pairs
            plt.figure(figsize=(12, 6))
            plt.bar(range(20), top_pairs['Similarity'])
            plt.xticks(range(20), [f"{row['Node1']}-{row['Node2']}" 
                                  for _, row in top_pairs.iterrows()],
                      rotation=45, ha='right')
            plt.title(f'Top 20 Most Similar Pairs ({method})')
            plt.tight_layout()
            plt.savefig(f"{self.output_dirs['visualizations']}/top_pairs_{method}.png")
            plt.close()
        
        # Đánh giá dự đoán
        results = self._evaluate_link_prediction(similarity_scores, test_edges)
        
        # Lưu kết quả đánh giá
        with open(f"{self.output_dirs['link_prediction']}/prediction_results.json", 'w', 
                  encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # Tạo bảng so sánh các phương pháp
        comparison_df = pd.DataFrame(results).T
        comparison_df.to_csv(f"{self.output_dirs['link_prediction']}/method_comparison.csv")
        
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

    def _evaluate_link_prediction(self, similarity_scores: Dict, test_edges: List) -> Dict:
        """Đánh giá kết quả dự đoán liên kết"""
        results = {}
        
        for method_name, score_dict in similarity_scores.items():
            # Lấy top K predictions (K = số lượng test edges)
            k = len(test_edges)
            predictions = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:k]
            predicted_edges = [edge for edge, _ in predictions]
            
            # Tính các metrics
            true_positives = sum(1 for edge in test_edges 
                               if edge in predicted_edges or edge[::-1] in predicted_edges)
            
            precision = true_positives / len(predicted_edges) if predicted_edges else 0
            recall = true_positives / len(test_edges) if test_edges else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Tính AUC
            all_scores = []
            all_labels = []
            
            # Tạo labels và scores cho tất cả các cặp có thể
            for (u, v), score in score_dict.items():
                all_scores.append(score)
                # Label = 1 nếu edge có trong test_edges
                label = 1 if (u, v) in test_edges or (v, u) in test_edges else 0
                all_labels.append(label)
                
            # Tính AUC
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(all_labels, all_scores)
            except:
                auc = 0.0
            
            results[method_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'number_of_predictions': len(predicted_edges),
                'true_positives': true_positives
            }
            
            # In kết quả chi tiết
            print(f"\nResults for {method_name}:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"True Positives: {true_positives}")
            print(f"Number of Predictions: {len(predicted_edges)}")
            
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