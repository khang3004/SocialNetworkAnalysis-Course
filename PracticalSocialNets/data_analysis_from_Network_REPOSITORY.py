import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import pandas as pd
import numpy as np
import zipfile
import time
import scipy
import io

def download_and_read_data():
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    try:
        
    response = urllib.request.urlopen(url= url)
    data = response.read()

    df = pd.read_csv(io.BytesIO(data),
                     compression= 'gzip',
                     header=None,
                     names = ['source','target'])
    return df
def calculate_all_metrics(G):
    """
        Tính toán tất cả các số đo của đồ thị
        Args:
            G: Đồ thị NetworkX
        Returns:
            dict: Dictionary chứa các số đo của đồ thị
    """
    metrics = {}
    # 1. Thông tin cơ bản
    metrics['nodes'] = G.number_of_nodes()
    # |V|: Số lượng đỉnh trong đồ thị

    metrics['edges'] = G.number_of_edges()
    # |E|: Số lượng cạnh trong đồ thị

    metrics['density'] = nx.density(G)
    # Mật độ đồ thị
    # Công thức: D = 2|E| / (|V|(|V|-1))
    # Trong đó:
    # - |E|: số cạnh thực tế
    # - |V|(|V|-1): số cạnh tối đa có thể có trong đồ thị vô hướng

    metrics['average_degree'] = np.mean([d for n, d in G.degree()])

    # 2. Degree metrics
    degrees = dict(G.degree())
    metrics['avg_degree'] = sum(degrees.values()) / len(degrees)
    # Degree trung bình
    # Công thức: <k> = (1/|V|) * Σ ki
    # Trong đó:
    # - |V|: số lượng đỉnh
    # - ki: degree của đỉnh i
    # Note: Trong đồ thị vô hướng: <k> = 2|E|/|V|

    metrics['max_degree'] = max(degrees.values())
    metrics['degrees'] = degrees

    # 3. Degree Centrality
    dc = nx.degree_centrality(G)
    metrics['degree_centrality'] = {
        'values': dc,
        'max': max(dc.values()),
        'avg': sum(dc.values()) / len(dc),
        'node_max': max(dc, key=dc.get)
    }
    # Degree Centrality
    # Công thức: CD(v) = deg(v)/(|V|-1)
    # Trong đó:
    # - deg(v): degree của đỉnh v
    # - |V|-1: số lượng kết nối tối đa có thể có của một đỉnh

    # 4. Betweenness Centrality
    bc = nx.betweenness_centrality(G)
    metrics['betweenness_centrality'] = {
        'values': bc,
        'max': max(bc.values()),
        'avg': sum(bc.values()) / len(bc),
        'node_max': max(bc, key=bc.get)
    }
    # Betweenness Centrality
    # Công thức: CB(v) = Σ (σst(v)/σst)
    # Trong đó:
    # - σst: số đường đi ngắn nhất từ đỉnh s đến đỉnh t
    # - σst(v): số đường đi ngắn nhất từ s đến t đi qua v
    # - Tổng được tính trên mọi cặp đỉnh s,t khác v

    # 5. Closeness Centrality
    cc = nx.closeness_centrality(G)
    metrics['closeness_centrality'] = {
        'values': cc,
        'max': max(cc.values()),
        'avg': sum(cc.values()) / len(cc),
        'node_max': max(cc, key=cc.get)
    }
    # Closeness Centrality
    # Công thức: CC(v) = (|V|-1) / Σ d(v,u)
    # Trong đó:
    # - |V|-1: số đỉnh khác v
    # - d(v,u): độ dài đường đi ngắn nhất từ v đến u
    # - Tổng được tính trên mọi đỉnh u khác v

    # 6. PageRank
    pr = nx.pagerank(G, alpha=0.85)
    metrics['pagerank'] = {
        'values': pr,
        'max': max(pr.values()),
        'avg': sum(pr.values()) / len(pr),
        'node_max': max(pr, key=pr.get)
    }
    # PageRank
    # Công thức: PR(v) = (1-d) + d * Σ (PR(u)/OutDegree(u))
    # Trong đó:
    # - d: damping factor (thường = 0.85)
    # - PR(u): PageRank của các đỉnh u kề với v
    # - OutDegree(u): bậc ra của đỉnh u
    # - Tổng được tính trên mọi đỉnh u kề với v

    return metrics

def print_detailed_results(metrics):
    """
    In kết quả chi tiết của các phép đo
    """
    print("\n======== PHÂN TÍCH MẠNG XÃ HỘI ========= ")
    print("1. Thông tin cơ bản")
    print(f"- Số lượng nodes (người dùng): {metrics['nodes']}")
    print(f"- Số lượng cạnh (kết nối): {metrics['edges']}")
    print(f"- Mật độ: {metrics['density']}")
    print(f"- Degree trung bình: {metrics['average_degree']}")
    print(f"- Degree lớn nhất: {metrics['max_degree']}")

    centrality_measures = {
            'Degree Centrality': 'degree_centrality',
            'Betweenness Centrality': 'betweenness_centrality',
            'Closeness Centrality': 'closeness_centrality',
            'PageRank': 'pagerank'
        }

    print("2. Các số đo Centrality")
    for name, measure in centrality_measures.items():
        print(f"- {name}")
        print(f"- Giá trị lớn nhất: {metrics[measure]['max']:.4f}")
        print(f"- Giá trị trung bình: {metrics[measure]['avg']:.4f}")
        print(f"- Node có giá trị cao nhất: {metrics[measure]['node_max']}")
    
def visualize_metrics(metrics):
    """
    Trực quan hóa các số đo mạng xã hội bằng các biểu đồ
    """
    plt.style.use('seaborn')
    
    # Tạo figure với 2x2 subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Biểu đồ phân phối degree
    plt.subplot(221)
    degree_values = list(metrics['degree_centrality']['values'].values())
    plt.hist(degree_values, bins=30, color='skyblue', edgecolor='black')
    plt.title('Phân phối Degree Centrality', fontsize=14, pad=15)
    plt.xlabel('Degree Centrality')
    plt.ylabel('Số lượng nodes')
    
    # 2. So sánh các centrality measures cho top 10 nodes
    plt.subplot(222)
    measures = {
        'Degree': metrics['degree_centrality']['values'],
        'Betweenness': metrics['betweenness_centrality']['values'],
        'Closeness': metrics['closeness_centrality']['values'],
        'PageRank': metrics['pagerank']['values']
    }
    
    # Lấy top 10 nodes theo PageRank
    top_nodes = sorted(metrics['pagerank']['values'].items(), 
                      key=lambda x: x[1], reverse=True)[:10]
    top_node_ids = [node[0] for node in top_nodes]
    
    x = np.arange(len(top_node_ids))
    width = 0.2
    multiplier = 0
    
    for measure_name, measure_values in measures.items():
        # Chuẩn hóa giá trị về khoảng [0,1]
        normalized_values = [measure_values[node]/max(measure_values.values()) 
                           for node in top_node_ids]
        offset = width * multiplier
        plt.bar(x + offset, normalized_values, width, label=measure_name)
        multiplier += 1
    
    plt.title('So sánh các Centrality Measures\ncho Top 10 Nodes', fontsize=14, pad=15)
    plt.xlabel('Node ID')
    plt.ylabel('Giá trị (đã chuẩn hóa)')
    plt.xticks(x + width * 1.5, top_node_ids, rotation=45)
    plt.legend(loc='upper right')
    
    # 3. Biểu đồ tròn thể hiện phân phối PageRank
    plt.subplot(223)
    top_pr = dict(sorted(metrics['pagerank']['values'].items(), 
                        key=lambda x: x[1], reverse=True)[:5])
    other_pr = sum(value for key, value in metrics['pagerank']['values'].items() 
                  if key not in top_pr)
    top_pr['Others'] = other_pr
    
    plt.pie(top_pr.values(), labels=top_pr.keys(), autopct='%1.1f%%',
            colors=plt.cm.Pastel1(np.linspace(0, 1, len(top_pr))),
            explode=[0.05] * len(top_pr))
    plt.title('Phân phối PageRank (Top 5 nodes)', fontsize=14, pad=15)
    
    # 4. Box plot cho tất cả các centrality measures
    plt.subplot(224)
    data = [list(measure['values'].values()) for measure in 
            [metrics['degree_centrality'],
             metrics['betweenness_centrality'],
             metrics['closeness_centrality'],
             metrics['pagerank']]]
    
    plt.boxplot(data, labels=['Degree', 'Betweenness', 'Closeness', 'PageRank'])
    plt.title('Phân phối các Centrality Measures', fontsize=14, pad=15)
    plt.ylabel('Giá trị')
    
    plt.tight_layout()
    plt.show()


def main():
    df = download_and_read_data()
    print(df.head())
    G = nx.from_edgelist(df, 'source', 'target')
    metrics = calculate_all_metrics(G)
    print(metrics)

if __name__ == 'main':
    main()