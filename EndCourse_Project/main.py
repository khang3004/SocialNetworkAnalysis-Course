from src.channel_finder import ChannelFinder
from src.video_analyzer import VideoAnalyzer
from src.network_builder import NetworkBuilder
import pandas as pd

def main():
    # Khởi tạo các đối tượng
    channel_finder = ChannelFinder()
    video_analyzer = VideoAnalyzer()
    network_builder = NetworkBuilder()

    print("\nBắt đầu phân tích video và xây dựng mạng lưới collab...")

    # Load danh sách kênh
    channels_df = pd.read_csv('output/official_channels.csv')
    total_channels = len(channels_df)
    print(f"\nBắt đầu phân tích {total_channels} kênh...")

    # Phân tích từng kênh và xây dựng mạng lưới
    for idx, row in channels_df.iterrows():
        channel_url = row['channel_url']
        artist_name = row['artist_name']
        
        print(f"\nĐang phân tích kênh {idx+1}/{total_channels}: {artist_name}")
        
        # Lấy danh sách video của kênh
        videos = video_analyzer.get_channel_videos(channel_url)
        print(f"Tìm thấy {len(videos)} video chính thức")
        
        # Phân tích collaborators trong từng video
        for video_url in videos:
            collaborators = video_analyzer.get_video_collaborators(video_url, artist_name)
            if collaborators:
                print(f"Tìm thấy collab với: {', '.join(collaborators)}")
                # Thêm vào mạng lưới
                network_builder.add_collaboration(artist_name, collaborators)

    # Lưu mạng lưới
    network_builder.save_network()

if __name__ == "__main__":
    main() 