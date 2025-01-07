from typing import Set, Dict
from yt_dlp import YoutubeDL
from ytmusicapi import YTMusic
import time
import pandas as pd
import os
from .utils import load_artists_from_csv

class ChannelFinder:
    def __init__(self):
        self.ytmusic = YTMusic()
        self.ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': True
        }
        self.channel_info = {}
        self.channel_cache = {}
        self.core_artists = load_artists_from_csv()
        
        # Tạo thư mục output nếu chưa có
        os.makedirs('output', exist_ok=True)
        
        # Khởi tạo file channels.csv nếu chưa tồn tại
        self.channels_file = 'output/official_channels.csv'
        if not os.path.exists(self.channels_file):
            # Tạo file mới với header
            pd.DataFrame(columns=[
                'artist_name', 
                'channel_name',
                'channel_url', 
                'channel_id'
            ]).to_csv(self.channels_file, index=False, encoding='utf-8')
            print(f"Đã tạo file {self.channels_file}")

    def search_channels(self) -> Set[str]:
        """Tìm kiếm kênh YouTube của nghệ sĩ"""
        found_channels = set()
        
        # Load danh sách nghệ sĩ đã có kênh từ cache
        try:
            existing_df = pd.read_csv(self.channels_file)
            cached_artists = set(existing_df['artist_name'].str.lower().str.strip())
            print(f"\nĐã load {len(cached_artists)} nghệ sĩ từ cache")
        except Exception:
            cached_artists = set()
            print("\nKhông tìm thấy cache, bắt đầu tìm kiếm mới")
        
        # Lọc ra những nghệ sĩ chưa có trong cache
        remaining_artists = set(artist.lower().strip() for artist in self.core_artists) - cached_artists
        print(f"\nCòn {len(remaining_artists)} nghệ sĩ cần tìm kênh...")
        
        # Tìm kiếm cho những nghệ sĩ còn lại
        for artist in self.core_artists:
            if artist.lower().strip() not in remaining_artists:
                continue
            
            try:
                print(f"\nĐang tìm kiếm kênh cho nghệ sĩ: {artist}")
                channels = self._search_from_ytdl(artist)
                if channels:
                    found_channels.update(channels)
                    self.save_channels_to_csv()
                time.sleep(1)  # Delay ngắn giữa các request
                
            except Exception as e:
                print(f"Lỗi khi tìm kênh cho {artist}: {str(e)}")
                continue
        
        print(f"\nĐã tìm thêm được {len(found_channels)} kênh verified mới")
        return found_channels

    def _search_from_ytdl(self, artist_name: str) -> Set[str]:
        """Tìm kênh từ YoutubeDL"""
        channels = set()
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                # Tìm kiếm với các query khác nhau để tăng khả năng tìm thấy kênh chính thức
                search_queries = [
                    f"ytsearch3:\"{artist_name} official\"",
                    f"ytsearch3:\"{artist_name} official channel\"", 
                    f"ytsearch3:\"{artist_name} official music\""
                ]

                for query in search_queries:
                    if channels:  # Nếu đã tìm thấy kênh thì dừng
                        break
                        
                    try:
                        results = ydl.extract_info(query, download=False)
                        if not results or 'entries' not in results:
                            continue

                        for entry in results['entries']:
                            if not entry:
                                continue
                                
                            channel_url = entry.get('uploader_url', '') or entry.get('channel_url', '')
                            channel_id = entry.get('channel_id', '')
                            channel_name = entry.get('uploader', '') or entry.get('channel', '')
                            
                            if not channel_url or not channel_id:
                                continue

                            # Kiểm tra trực tiếp kênh để xác minh là kênh chính thức
                            try:
                                channel_info = ydl.extract_info(channel_url, download=False)
                                if channel_info:
                                    # Chỉ lấy kênh verified hoặc official artist channel
                                    is_verified = channel_info.get('channel_is_verified', False)
                                    is_official = channel_info.get('channel_type') == 'official_artist_channel'
                                    
                                    if is_verified or is_official:
                                        channels.add(channel_url)
                                        self.channel_info[channel_url] = {
                                            'name': channel_name,
                                            'artist': artist_name,
                                            'verified': True,
                                            'id': channel_id
                                        }
                                        print(f"\nTìm thấy kênh chính thức: {channel_name}")
                                        return channels  # Return ngay khi tìm thấy kênh chính thức
                            except:
                                continue

                    except Exception as e:
                        print(f"Lỗi khi tìm với query '{query}': {str(e)}")
                        continue

                    time.sleep(0.5)  # Delay ngắn giữa các query
                
            return channels
            
        except Exception as e:
            print(f"Lỗi khi tìm kênh cho {artist_name}: {str(e)}")
            return channels

    def save_channels_to_csv(self):
        """Lưu danh sách kênh chính thức vào CSV"""
        try:
            if not self.channel_info:
                return

            # Đọc dữ liệu hiện có
            try:
                existing_df = pd.read_csv(self.channels_file)
            except:
                existing_df = pd.DataFrame(columns=[
                    'artist_name', 
                    'channel_name',
                    'channel_url', 
                    'channel_id'
                ])
            
            # Tạo DataFrame mới
            new_data = []
            for channel_url, info in self.channel_info.items():
                new_data.append({
                    'artist_name': info['artist'],
                    'channel_name': info['name'],
                    'channel_url': channel_url,
                    'channel_id': info['id']
                })
            new_df = pd.DataFrame(new_data)
            
            # Gộp và loại bỏ trùng lặp
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df.drop_duplicates(subset=['channel_url', 'artist_name'])
            
            # Sắp xếp theo tên nghệ sĩ
            combined_df = combined_df.sort_values('artist_name')
            
            # Lưu vào file
            combined_df.to_csv(self.channels_file, index=False, encoding='utf-8')
            print(f"\nĐã cập nhật file với tổng cộng {len(combined_df)} kênh")
            
        except Exception as e:
            print(f"Lỗi khi lưu channels: {str(e)}")