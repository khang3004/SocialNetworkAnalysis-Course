import json
from typing import Dict, Set
import os
from datetime import datetime

class DataSaver:
    """Class lưu trữ dữ liệu"""
    
    @staticmethod
    def save_data(network: Dict[str, Set[str]], channel_info: Dict):
        """Lưu dữ liệu vào file JSON"""
        try:
            # Tạo thư mục output nếu chưa tồn tại
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Thêm timestamp vào tên file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. Lưu thông tin kênh
            channel_data = {
                url: {
                    'name': info['name'],
                    'artist': info['artist'],
                    'verified': info.get('verified', False),
                    'id': info.get('id', '')
                }
                for url, info in channel_info.items()
            }
            
            channel_file = os.path.join(output_dir, f'channel_data_{timestamp}.json')
            with open(channel_file, 'w', encoding='utf-8') as f:
                json.dump(channel_data, f, ensure_ascii=False, indent=2)
                
            # 2. Lưu dữ liệu mạng lưới
            # Convert set thành list để có thể serialize
            graph_data = {
                artist: list(collaborators)
                for artist, collaborators in network.items()
            }
            
            network_file = os.path.join(output_dir, f'network_data_{timestamp}.json')
            with open(network_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            
            # 3. Lưu metadata
            metadata = {
                'timestamp': timestamp,
                'total_artists': len(network),
                'total_channels': len(channel_info),
                'total_collaborations': sum(len(collab) for collab in network.values()) // 2
            }
            
            metadata_file = os.path.join(output_dir, f'metadata_{timestamp}.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            print(f"\nĐã lưu dữ liệu vào thư mục {output_dir}:")
            print(f"- Thông tin kênh: {channel_file}")
            print(f"- Dữ liệu mạng lưới: {network_file}")
            print(f"- Metadata: {metadata_file}")
            
        except Exception as e:
            print(f"\nLỗi khi lưu dữ liệu: {str(e)}")
            
    @staticmethod
    def load_data(timestamp: str = None) -> tuple[Dict, Dict]:
        """Load dữ liệu từ file JSON"""
        try:
            output_dir = 'output'
            
            # Nếu không có timestamp, lấy file mới nhất
            if timestamp is None:
                files = os.listdir(output_dir)
                timestamps = [f.split('_')[1].split('.')[0] for f in files if f.startswith('network_data_')]
                if not timestamps:
                    raise FileNotFoundError("Không tìm thấy file dữ liệu")
                timestamp = max(timestamps)
            
            # Load network data
            network_file = os.path.join(output_dir, f'network_data_{timestamp}.json')
            with open(network_file, 'r', encoding='utf-8') as f:
                network_data = json.load(f)
                # Convert list back to set
                network_data = {k: set(v) for k, v in network_data.items()}
            
            # Load channel data
            channel_file = os.path.join(output_dir, f'channel_data_{timestamp}.json')
            with open(channel_file, 'r', encoding='utf-8') as f:
                channel_data = json.load(f)
            
            return network_data, channel_data
            
        except Exception as e:
            print(f"\nLỗi khi load dữ liệu: {str(e)}")
            return {}, {} 