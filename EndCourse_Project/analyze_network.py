from src.network_analyzer import NetworkAnalyzer

def main():
    # Khởi tạo analyzer
    analyzer = NetworkAnalyzer()
    
    print("Analyzing VPop collaboration network...")
    
    # 1. Phân tích metrics cơ bản
    print("\nAnalyzing basic metrics...")
    basic_metrics = analyzer.analyze_basic_metrics()
    
    # 2. Phân tích centrality
    print("\nAnalyzing centrality metrics...")
    centrality_metrics = analyzer.analyze_centrality_metrics()
    
    # 3. Phát hiện cộng đồng
    print("\nDetecting communities...")
    community_results = analyzer.detect_communities()
    
    # 4. Dự đoán liên kết
    print("\nPredicting links...")
    prediction_results = analyzer.predict_links()
    
    # 5. Trực quan hóa
    print("\nVisualizing network...")
    analyzer.visualize_network()
    
    print("\nAnalysis complete! Results saved in output directory.")

if __name__ == "__main__":
    main() 