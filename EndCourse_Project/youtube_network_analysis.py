from collections import defaultdict
import json
import time
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Dict, List
from yt_dlp import YoutubeDL
from ytmusicapi import YTMusic
from yt_dlp import YoutubeDL
import networkx as nx
from tqdm import tqdm
import time
import json
import os
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class YouTubeNetworkAnalyzer:
    def __init__(self):
        self.ytmusic = YTMusic()
        self.ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': True
        }
        self.G = nx.Graph()
        self.channel_info = {}
        self.channel_cache = {}
        
        # Đọc danh sách nghệ sĩ từ CSV thay vì hardcode
        try:
            import pandas as pd
            df = pd.read_csv('Vietnamese_Artists_Unique.csv')
            self.core_artists = set(df['Artist'].str.strip().unique())
            print(f"Đã đọc {len(self.core_artists)} nghệ sĩ từ CSV")
        except Exception as e:
            print(f"Lỗi khi đọc file CSV: {str(e)}")
            self.core_artists = set()  # Fallback to empty set if CSV read fails

    def search_channels(self) -> Set[str]:
        """Tìm kiếm thông minh các kênh YouTube của nghệ sĩ VPop"""
        found_channels = set()
        processed_artists = set()       
        
        try:
            # 1. Lấy danh sách nghệ sĩ từ YouTube Music Charts và Trending
            artists = self._get_artists_from_charts()
            print(f"\nĐã tìm thấy {len(artists)} nghệ sĩ từ charts")
            
            # 2. Tìm kiếm song song kênh của các nghệ sĩ
            with ThreadPoolExecutor(max_workers=5) as executor:
                def process_artist(artist):
                    if artist.lower() not in processed_artists:
                        processed_artists.add(artist.lower())
                        return self._search_artist_channel(artist)
                    return set()
                
                # Xử lý song song và gộp kết quả
                results = executor.map(process_artist, artists)
                for channels in results:
                    found_channels.update(channels)
            
            print(f"\nĐã tìm th���y {len(found_channels)} kênh verified")
            return found_channels
            
        except Exception as e:
            print(f"Lỗi khi tìm kiếm kênh: {str(e)}")
            return found_channels

    def _get_artists_from_charts(self) -> Set[str]:
        """Lấy danh sách nghệ sĩ từ YouTube Music"""
        artists = set()
        
        # Hàm helper để xử lý an toàn
        def extract_artists_from_item(item):
            if isinstance(item, dict):
                # Trường hợp 1: artists là list trong dict
                if 'artists' in item and isinstance(item['artists'], list):
                    for artist in item['artists']:
                        if isinstance(artist, dict) and 'name' in artist:
                            if self._is_vpop_artist(artist['name']):
                                artists.add(artist['name'])
                
                # Trường hợp 2: artist là dict trc tiếp
                elif 'artist' in item and isinstance(item['artist'], dict):
                    if 'name' in item['artist']:
                        if self._is_vpop_artist(item['artist']['name']):
                            artists.add(item['artist']['name'])
                
                # Trường hợp 3: artist là string
                elif 'artist' in item and isinstance(item['artist'], str):
                    if self._is_vpop_artist(item['artist']):
                        artists.add(item['artist'])
        
        try:
            # 1. Thêm core artists làm seed
            artists.update(self.core_artists)
            print(f"\nĐã thêm {len(self.core_artists)} nghệ sĩ core")

            # 2. Tìm từ YouTube Music Charts
            try:
                charts = self.ytmusic.get_charts('VN')
                if isinstance(charts, dict):
                    for chart_type in ['trending', 'songs', 'videos', 'artists']:
                        if chart_type in charts:
                            items = charts[chart_type]
                            if isinstance(items, list):
                                for item in items:
                                    extract_artists_from_item(item)
            except Exception as e:
                print(f"Lỗi khi lấy charts: {str(e)}")

            # 3. Tìm kiếm với nhiều từ khóa khác nhau
            search_queries = [
                # Thể loại
                'vpop', 'nhạc trẻ việt nam', 'underground việt', 
                'indie việt nam', 'rap việt', 'pop việt nam',
                
                # Xu hướng
                'trending vietnam music', 'viral vietnam', 'top ca sĩ việt'
            ]
            
            for query in search_queries:
                try:
                    # Tìm theo filter songs
                    results = self.ytmusic.search(query, filter='songs', limit=50)
                    if isinstance(results, list):
                        for song in results:
                            if isinstance(song, dict):
                                # Xử lý artists từ bài hát
                                if 'artists' in song and isinstance(song['artists'], list):
                                    for artist in song['artists']:
                                        if isinstance(artist, dict) and 'name' in artist:
                                            artist_name = artist['name'].strip()
                                            if len(artist_name) > 1:
                                                artists.add(artist_name)
                                
                                # Xử lý album artists nếu có
                                if 'album' in song and isinstance(song['album'], dict):
                                    if 'artists' in song['album']:
                                        for album_artist in song['album']['artists']:
                                            if isinstance(album_artist, dict) and 'name' in album_artist:
                                                artist_name = album_artist['name'].strip()
                                                if len(artist_name) > 1:
                                                    artists.add(artist_name)
                
                    # Tìm theo filter videos
                    video_results = self.ytmusic.search(query, filter='videos', limit=50)
                    if isinstance(video_results, list):
                        for video in video_results:
                            if isinstance(video, dict):
                                if 'artists' in video and isinstance(video['artists'], list):
                                    for artist in video['artists']:
                                        if isinstance(artist, dict) and 'name' in artist:
                                            artist_name = artist['name'].strip()
                                            if len(artist_name) > 1:
                                                artists.add(artist_name)
                
                    print(f"\nĐã tìm thấy nghệ sĩ từ query: {query}")
                    
                except Exception as e:
                    print(f"Lỗi khi search với query {query}: {str(e)}")
                    continue
                    
                time.sleep(1)  # Tránh rate limit

            # 3. Tìm kiếm related artists
            found_artists = list(artists)[:30]  # Lấy 30 ngh sĩ đầu làm seed
            for artist_name in found_artists:
                try:
                    # Tìm related artists
                    results = self.ytmusic.search(f"{artist_name} similar artists", filter='artists', limit=10)
                    if isinstance(results, list):
                        for artist in results:
                            if isinstance(artist, dict):
                                if 'artist' in artist and isinstance(artist['artist'], dict):
                                    if 'name' in artist['artist']:
                                        artist_name = artist['artist']['name'].strip()
                                        if len(artist_name) > 1:
                                            artists.add(artist_name)
                
                    print(f"\nĐã tìm thấy nghệ sĩ liên quan từ: {artist_name}")
                    
                except Exception as e:
                    continue
                    
                time.sleep(1)

            # 4. Lọc và làm sạch dữ liệu
            artists = {a for a in artists if isinstance(a, str) and len(a.strip()) > 1}
            
            print(f"\nTổng cộng đã tìm thấy {len(artists)} nghệ sĩ")
            return artists
                
        except Exception as e:
            print(f"Lỗi tổng thể khi lấy danh sách nghệ sĩ: {str(e)}")
            return self.core_artists

    def get_channel_videos(self, channel_url: str, min_videos: int = 10) -> List[str]:
        """Lấy danh sách video chính thức của kênh"""
        videos = []
        
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                # Thêm error handling
                if not channel_url or 'youtube.com' not in channel_url:
                    return videos
                    
                # Lấy playlist "Uploads" của kênh
                try:
                    channel_info = ydl.extract_info(channel_url, download=False)
                    if not channel_info or 'id' not in channel_info:
                        return videos
                        
                    uploads_playlist = f"UU{channel_info['id'][2:]}"
                    playlist_info = ydl.extract_info(
                        f"https://www.youtube.com/playlist?list={uploads_playlist}",
                        download=False
                    )
                    
                    if playlist_info and 'entries' in playlist_info:
                        for entry in playlist_info['entries']:
                            if not entry:
                                continue
                                
                            # Kiểm tra null trước khi access
                            title = entry.get('title', '')
                            description = entry.get('description', '')
                            
                            if title and description:  # Chỉ xử lý khi có data
                                title = title.lower()
                                description = description.lower()
                                
                                if self._is_official_video(title, description):
                                    video_url = entry.get('webpage_url', '')
                                    if video_url and video_url not in videos:
                                        videos.append(video_url)
                                        print(f"\nTìm thấy video chính thức: {title}")
                                        
                                if len(videos) >= min_videos:
                                    break
                                    
                except Exception as e:
                    print(f"\nLỗi khi lấy playlist của {channel_url}: {str(e)}")
                    return videos
                    
        except Exception as e:
            print(f"\nLỗi khi lấy video của {channel_url}: {str(e)}")
            
        return videos

    def _is_official_video(self, title: str, description: str) -> bool:
        """Kiểm tra xem video có phải là MV chính thức không"""
        title = title.lower()
        description = description.lower()
        
        # Các từ khóa chỉ video chính thức
        official_keywords = {
            'official mv', 'official music video', 
            'official lyric', 'official visualizer',
            'm/v', 'mv official', 'lyric video',
            'lyrics video', 'visualizer'
        }
        
        # Các từ khóa loại trừ
        exclude_keywords = {
            'behind the scenes', 'making', 'karaoke', 
            'live', 'concert', 'dance practice', 'cover',
            'reaction', f'teaser', 'trailer', 'preview',
            'talkshow', 'interview', 'gameshow', 'vlog',
            'shorts', 'tiktok', 'rehearsal', 'practice',
            'fanmade', 'fan made', 'unofficial'
        }
        
        return (
            any(k in title for k in official_keywords) and
            not any(k in title for k in exclude_keywords)
        )

    def _is_vpop_artist(self, artist_name: str) -> bool:
        """Kiểm tra xem có phải nghệ sĩ VPop không"""
        if not isinstance(artist_name, str) or len(artist_name.strip()) < 2:
            return False
            
        artist_name = artist_name.lower().strip()
        
        # 1. Danh sách nghệ sĩ Vpop đã xác nhận
        verified_vpop_artists = {
            # Mainstream
            'son tung mtp', 'hoang thuy linh', 'den vau', 'jack', 'binz', 'min', 
            'duc phuc', 'erik', 'chi pu', 'karik', 'suboi', 'rhymastic', 'justatee',
            'soobin hoang son', 'bich phuong', 'my tam', 'noo phuoc thinh',
            'ha anh tuan', 'toc tien', 'dong nhi', 'vu cat tuong', 'huong tram',
            'ho ngoc ha', 'dam vinh hung', 'le quyen', 'tuan hung', 'lam truong',
            'dan truong', 'cam ly', 'quang dung', 'bang kieu', 'phuong thanh',
            'thu minh', 'uyen linh', 'hong nhung', 'thanh lam', 'trong tan',
            
            # New Gen
            'tlinh', 'mck', 'wren evans', 'mono', 'hieuthuhai', 'grey d',
            'phan manh quynh', 'vu', 'hoang dung', 'amee', 'han sara', 
            'juky san', 'orange', 'hoang thuy', 'lyly', 'phuc du', 'obito',
            'wxrdie', 'sol7', 'rpt mck', 'low g', 'vsoul', 'kien trung',
            
            # Underground/Indie
            'da lab', 'ngot', 'chillies', 'madihu', 'trang', '7uppercuts',
            'lil wuyn', 'manbo', 'thieu bao tram', 'vu thanh van', 'tam ka pkg',
            'b ray', 'young h', 'andree', 'lena', 'mai ngô', 'gill', 
            
            # Producer/Composer
            'huy tuan', 'duc tri', 'khac hung', 'do hieu', 'nguyen hong thuan',
            'chau dang khoa', 'triple d', 'onionn', 'masew', 'dtap', 'hua kim tuyen',
            'hoai sa', 'tien cookie', 'long halo', 'viruss', 'hoaprox'
        }
        
        # 2. Kiểm tra tên nghệ sĩ đã verify
        if any(name in artist_name for name in verified_vpop_artists):
            return True
        
        # 3. Kiểm tra tên Việt Nam phổ biến (họ)
        viet_surnames = {
            'nguyen', 'tran', 'le', 'pham', 'hoang', 'huynh', 'phan', 'vu', 'vo',
            'dang', 'bui', 'do', 'ho', 'ngo', 'duong', 'ly', 'dinh', 'cao', 'mai',
            'truong', 'lam', 'luu', 'trinh', 'ta', 'phung', 'dao'
        }
        
        # 4. Kiểm tra tên Việt Nam phổ biến (tên)
        viet_names = {
            'tung', 'hung', 'duc', 'minh', 'quang', 'thanh', 'tuan', 'dat', 'huy',
            'linh', 'thuy', 'huong', 'trang', 'mai', 'lan', 'phuong', 'thu', 'ha',
            'anh', 'hong', 'hoa', 'thao', 'nhung', 'dung', 'van', 'tam', 'tien',
            'trung', 'thi', 'hieu', 'phuc', 'thang', 'son', 'long', 'nam', 'binh'
        }
        
        name_parts = artist_name.split()
        has_viet_name = (
            any(part in viet_surnames for part in name_parts) or
            any(part in viet_names for part in name_parts)
        )
        
        # 5. Kiểm tra stage name tiếng Việt
        viet_stage_keywords = {
            'tung', 'den', 'binz', 'karik', 'wowy', 'suboi', 'erik', 'min', 'mono',
            'tlinh', 'mck', 'wxrdie', 'sol7', 'gill', 'rpt', 'soobin', 'amee',
            'juky', 'grey d', 'obito', 'vsoul', 'andree', 'masew', 'onionn', 'dtap'
        }
        
        has_viet_stage = any(keyword in artist_name for keyword in viet_stage_keywords)
        
        # 6. Kiểm tra từ khóa chỉ nghệ sĩ Việt
        vpop_keywords = {
            'vpop', 'v-pop', 'việt nam', 'vietnam', 'vietnamese',
            'underground việt', 'rap việt', 'indie việt'
        }
        
        has_vpop_keyword = any(keyword in artist_name for keyword in vpop_keywords)
        
        # 7. Loại bỏ các từ khóa nước ngoài
        foreign_keywords = {
            'kpop', 'k-pop', 'jpop', 'j-pop', 'cpop', 'c-pop',
            'korean', 'japanese', 'chinese', 'thai', 'english',
            'us', 'uk', 'american', 'british', 'international'
        }
        
        if any(keyword in artist_name for keyword in foreign_keywords):
            return False
            
        # 8. Kiểm tra dấu tiếng Việt
        vietnamese_chars = set('áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
        has_vietnamese = any(c in vietnamese_chars for c in artist_name)
        
        # Kết hợp các điều kiện
        return (artist_name in verified_vpop_artists) or \
               (has_viet_name and not any(keyword in artist_name for keyword in foreign_keywords)) or \
               (has_viet_stage and not any(keyword in artist_name for keyword in foreign_keywords)) or \
               (has_vietnamese and not any(keyword in artist_name for keyword in foreign_keywords)) or \
               (has_vpop_keyword and not any(keyword in artist_name for keyword in foreign_keywords))

    def _search_artist_channel(self, artist_name: str) -> Set[str]:
        """Tìm kênh YouTube chính thức của nghệ sĩ"""
        channels = set()
        processed_channels = set()
        
        try:
            # 1. Kiểm tra cache trước
            cache_key = artist_name.lower().strip()
            if cache_key in self.channel_cache:
                return self.channel_cache[cache_key]

            # 2. Tìm kiếm với nhiều biến thể tên
            name_variants = [
                artist_name,
                f"{artist_name} Official",
                f"{artist_name} OFFICIAL",
                f"{artist_name} Chính Thức",
                f"{artist_name} Music",
                f"Official {artist_name}",
                f"{artist_name} Entertainment",
                f"{artist_name} TV",
                f"{artist_name} Channel",
                # Thêm các biến thể viết thường
                artist_name.lower(),
                f"{artist_name.lower()} official",
                f"official {artist_name.lower()}"
            ]
            
            for query in name_variants:
                if channels:  # Nếu đã tìm thấy thì dừng
                    break
                    
                try:
                    with YoutubeDL(self.ydl_opts) as ydl:
                        # Tăng số lượng kết quả tìm kiếm
                        results = ydl.extract_info(f"ytsearch5:{query}", download=False)
                        
                        if results and 'entries' in results:
                            for entry in results['entries']:
                                if not entry:
                                    continue
                                    
                                channel_url = entry.get('channel_url', '')
                                channel_id = entry.get('channel_id', '')
                                channel_name = entry.get('channel', '')
                                
                                if not channel_url or not channel_id or channel_id in processed_channels:
                                    continue
                                    
                                processed_channels.add(channel_id)
                                
                                # Cải thiện việc kiểm tra tên kênh
                                if self._is_valid_channel_name(channel_name, artist_name):
                                    if self._verify_channel_content(channel_url, artist_name):
                                        channels.add(channel_url)
                                        self.channel_info[channel_url] = {
                                            'name': channel_name,
                                            'artist': artist_name,
                                            'verified': True,
                                            'id': channel_id
                                        }
                                        print(f"\nTìm thấy kênh chính thức: {channel_name}")
                                        break
                            
                except Exception as e:
                    print(f"\nLỗi khi tìm kiếm với query '{query}': {str(e)}")
                    continue
                    
                time.sleep(0.5)  # Tránh rate limit

            # Lưu vào cache
            self.channel_cache[cache_key] = channels
            return channels
                
        except Exception as e:
            print(f"\nLỗi khi tìm kênh của {artist_name}: {str(e)}")
            return channels

    def _is_valid_channel_name(self, channel_name: str, artist_name: str) -> bool:
        """Kiểm tra tên kênh có hợp lệ không"""
        if not isinstance(channel_name, str) or not isinstance(artist_name, str):
            return False
        
        channel_name = channel_name.lower().strip()
        artist_name = artist_name.lower().strip()
        
        # 1. Loại bỏ các kênh không liên quan
        invalid_keywords = {
            'fan', 'club', 'fc', 'community', 'cover', 'reaction',
            'gaming', 'game', 'play', 'stream', 'live', 'radio',
            'news', 'entertainment', 'media', 'network', 'tv',
            'vlog', 'daily', 'life', 'family', 'kids', 'beauty',
            'food', 'travel', 'sports', 'fitness', 'tutorial',
            'review', 'unbox', 'shop', 'store', 'official store'
        }
        
        if any(keyword in channel_name for keyword in invalid_keywords):
            return False
        
        # 2. Kiểm tra các từ khóa nước ngoài
        foreign_keywords = {
            'kpop', 'jpop', 'cpop', 'thai', 'korean', 'japanese', 'chinese',
            'english', 'american', 'british', 'indian', 'bollywood',
            'spanish', 'latin', 'french', 'german', 'russian',
            'international', 'world', 'global'
        }
        
        if any(keyword in channel_name for keyword in foreign_keywords):
            return False
        
        # 3. Kiểm tra tên nghệ sĩ có trong tên kênh
        name_parts = artist_name.split()
        if not any(part in channel_name for part in name_parts):
            return False
        
        # 4. Kiểm tra các ký tự đặc biệt và số lượng
        special_chars = set('!@#$%^&*()+=[]{}|\\:;"\'<>?,/')
        if len([c for c in channel_name if c in special_chars]) > 2:
            return False
        
        return True

    def _is_official_channel(self, channel_name: str, artist_name: str, channel_id: str = None) -> bool:
        """Kiểm tra xem có ph��i kênh chính thức không"""
        if not isinstance(channel_name, str) or not isinstance(artist_name, str):
            return False
        
        channel_name = channel_name.lower().strip()
        artist_name = artist_name.lower().strip()

        try:
            # 1. Kiểm tra tên kênh có chứa tên nghệ sĩ không
            artist_name_parts = set(artist_name.split())
            channel_name_parts = set(channel_name.split())
            
            # Cần có ít nhất một phần của tên nghệ sĩ trong tên kênh
            if not any(part in channel_name for part in artist_name_parts):
                return False

            # 2. Kiểm tra các từ khóa chỉ kênh chính thức
            official_keywords = {
                'official', 'chính thức', 'verified', 'music channel',
                'artist channel', 'musician'
            }
            
            has_official_keyword = any(keyword in channel_name for keyword in official_keywords)
            if not has_official_keyword:
                return False

            # 3. Kiểm tra cc từ khóa chỉ kênh không chính thức
            unofficial_keywords = {
                'fan', 'club', 'fc', 'community', 'cover', 'reaction',
                'tribute', 'best of', 'collection', 'mix', 'playlist',
                'top hits', 'karaoke', 'instrumental', 'backing track',
                'gaming', 'game', 'play', 'stream', 'radio', 'news',
                'vlog', 'daily', 'life', 'family', 'beauty', 'food'
            }
            
            if any(keyword in channel_name for keyword in unofficial_keywords):
                return False

            # 4. Kiểm tra metadata của kênh qua YouTube Data API (nếu có channel_id)
            if channel_id:
                try:
                    with YoutubeDL(self.ydl_opts) as ydl:
                        channel_info = ydl.extract_info(
                            f"https://www.youtube.com/channel/{channel_id}", 
                            download=False
                        )
                        
                        if channel_info:
                            # Kiểm tra verified badge nếu có
                            is_verified = channel_info.get('verified', False)
                            if is_verified:
                                return True
                            
                            # Kiểm tra subscriber count
                            subscriber_count = channel_info.get('subscriber_count', 0)
                            if subscriber_count > 50000:  # Giảm ngưỡng xuống
                                return True
                            
                            # Kiểm tra channel description
                            description = channel_info.get('description', '').lower()
                            music_keywords = {
                                'ca sĩ', 'nghệ sĩ', 'rapper', 'producer', 'nhạc sĩ',
                                'musician', 'artist', 'singer', 'composer'
                            }
                            if any(keyword in description for keyword in music_keywords):
                                return True
                                
                except Exception:
                    # Nếu không lấy được metadata, chỉ dựa vào tên kênh
                    pass

            # 5. Kiểm tra format tên chuẩn
            standard_formats = [
                f"{artist_name} official",
                f"official {artist_name}",
                f"{artist_name} music",
                f"{artist_name} chính thức"
            ]
            
            if any(format in channel_name for format in standard_formats):
                return True

            # 6. Kiểm tra tên kênh có dấu tích xanh (✓ hoặc ✔)
            if any(mark in channel_name for mark in ['✓', '✔']):
                return True

            return False
            
        except Exception as e:
            print(f"Lỗi khi kiểm tra kênh {channel_name}: {str(e)}")
            return False

    def _process_playlist(self, playlist: Dict, artists: Set[str]) -> None:
        """Xử lý playlist để tìm nghệ sĩ"""
        try:
            playlist_id = playlist.get('playlistId')
            if not playlist_id:
                return
            
            # Lấy thông tin chi tiết playlist
            playlist_items = self.ytmusic.get_playlist(playlist_id, limit=100)
            for track in playlist_items.get('tracks', []):
                if 'artists' in track:
                    for artist in track['artists']:
                        if self._is_vpop_artist(artist['name']):
                            artists.add(artist['name'])
                            
        except Exception:
            pass

    def get_video_collaborators(self, video_url: str, channel_artist: str) -> Set[str]:
        """Lấy danh sách nghệ sĩ collab trong một video"""
        collaborators = set()
        
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                video_info = ydl.extract_info(video_url, download=False)
                
                if not video_info:
                    return collaborators
                    
                # 1. Xử lý title video
                title = video_info.get('title', '').lower()
                description = video_info.get('description', '').lower()
                
                # Các từ khóa chỉ collab
                collab_keywords = [
                    'ft.', 'ft', 'feat.', 'feat', 'featuring', 'với', 'with',
                    'x', 'vs.', 'vs', 'cùng', 'và', '&', 'prod.', 'prod by',
                    'produced by', 'beat by', 'mix by', 'mixed by', 'master by',
                    'mastered by', 'instrumental by', 'compose by', 'composed by',
                    'music by', 'lyrics by', 'written by', 'arrangement by',
                    'arranged by', 'remix by', 'cover by'
                ]
                
                # 2. Tìm nghệ sĩ từ title
                for keyword in collab_keywords:
                    if keyword in title:
                        # Tách phần sau keyword
                        parts = title.split(keyword)
                        if len(parts) > 1:
                            # Xử lý phần chứa tên nghệ sĩ
                            artist_part = parts[1].strip()
                            # Tách theo dấu phẩy, &, x, và
                            for separator in [',', '&', 'x', 'và']:
                                if separator in artist_part:
                                    potential_artists = [a.strip() for a in artist_part.split(separator)]
                                    for artist in potential_artists:
                                        if self._is_vpop_artist(artist) and artist.lower() != channel_artist.lower():
                                            collaborators.add(artist)
                                    break
                            else:
                                # Nếu không có separator
                                if self._is_vpop_artist(artist_part) and artist_part.lower() != channel_artist.lower():
                                    collaborators.add(artist_part)
                
                # 3. Tìm nghệ sĩ từ description
                desc_lines = description.split('\n')
                credit_keywords = [
                    'credit', 'credits', 'info', 'information', 'thông tin',
                    'performer', 'ca sĩ', 'vocalist', 'artist', 'nghệ sĩ',
                    'producer', 'beat maker', 'nhạc sĩ', 'composer', 'sáng tác',
                    'songwriter', 'author', 'lyrics', 'lyricist', 'tác giả',
                    'rap', 'rapper', 'featuring', 'collab', 'hợp tác',
                    'mixing', 'master', 'hoà âm', 'phối khí', 'arrangement'
                ]
                
                for line in desc_lines:
                    line = line.lower().strip()
                    if any(keyword in line for keyword in credit_keywords):
                        # Tìm tên nghệ sĩ sau dấu :, -, =
                        for separator in [':', '-', '=']:
                            if separator in line:
                                artist_part = line.split(separator)[1].strip()
                                # Tách nhiều nghệ sĩ
                                for sub_separator in [',', '&', 'x', 'và']:
                                    if sub_separator in artist_part:
                                        potential_artists = [a.strip() for a in artist_part.split(sub_separator)]
                                        for artist in potential_artists:
                                            if self._is_vpop_artist(artist) and artist.lower() != channel_artist.lower():
                                                collaborators.add(artist)
                                        break
                                else:
                                    if self._is_vpop_artist(artist_part) and artist_part.lower() != channel_artist.lower():
                                        collaborators.add(artist_part)
                
                # 4. Tìm từ tags
                tags = video_info.get('tags', [])
                if tags:
                    for tag in tags:
                        tag = tag.lower().strip()
                        if any(keyword in tag for keyword in collab_keywords):
                            potential_artist = tag.split(keyword)[1].strip() if keyword in tag else tag
                            if self._is_vpop_artist(potential_artist) and potential_artist.lower() != channel_artist.lower():
                                collaborators.add(potential_artist)
                
                # 5. Làm sạch tên nghệ sĩ
                collaborators = {c.strip() for c in collaborators if len(c.strip()) > 1}
                
                # 6. Loại bỏ chính nghệ sĩ của kênh và các biến thể tên
                channel_artist_lower = channel_artist.lower()
                collaborators = {c for c in collaborators 
                               if c.lower() != channel_artist_lower 
                               and channel_artist_lower not in c.lower()
                               and c.lower() not in channel_artist_lower}
                
        except Exception as e:
            print(f"\nLỗi khi lấy collaborators từ {video_url}: {str(e)}")
            
        return collaborators

    def analyze_network(self) -> None:
        """Phân tích mạng lưới nghệ sĩ"""
        try:
            # 1. Tìm kênh YouTube của các nghệ sĩ
            channels = self.search_channels()
            if not channels:
                print("Không tìm thấy kênh nào!")
                return
            
            # 2. Lấy video từ mỗi kênh và tìm collab
            network = defaultdict(set)
            
            for channel_url, info in self.channel_info.items():
                channel_artist = info['artist']  # Lấy tên nghệ sĩ của kênh
                try:
                    # Lấy danh sách video từ kênh
                    videos = self.get_channel_videos(channel_url)
                    
                    # Tìm nghệ sĩ collab trong mỗi video
                    for video_url in videos:
                        try:
                            collaborators = self.get_video_collaborators(
                                video_url=video_url,
                                channel_artist=channel_artist  # Truyền tên nghệ sĩ của kênh
                            )
                            
                            # Thêm vào mng lưới
                            if collaborators:
                                network[channel_artist].update(collaborators)
                                # Thêm cả chiều ngược lại
                                for collaborator in collaborators:
                                    network[collaborator].add(channel_artist)
                                    
                        except Exception as e:
                            print(f"\nLỗi khi xử lý video {video_url}: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"\nLỗi khi xử lý kênh {channel_url}: {str(e)}")
                    continue
                    
                time.sleep(1)  # Tránh rate limit

            # 3. Lưu dữ liệu
            self.save_data(network)
            
            # 4. Phân tích metrics
            if network:
                self.analyze_metrics(network)
                # 5. Trực quan hóa
                self.visualize_network(network)
            else:
                print("Không có dữ liệu để phân tích!")
                
        except Exception as e:
            print(f"\nLỗi khi phân tích mạng lưới: {str(e)}")

    def save_data(self, network: Dict[str, Set[str]]):
        """Lưu dữ liệu vào file JSON"""
        try:
            # 1. Lưu thông tin kênh
            channel_data = {
                url: {
                    'name': info['name'],
                    'artist': info['artist'],
                    'verified': info.get('verified', False)
                }
                for url, info in self.channel_info.items()
            }
            
            with open('channel_data.json', 'w', encoding='utf-8') as f:
                json.dump(channel_data, f, ensure_ascii=False, indent=2)
                
            # 2. Lưu dữ liệu mng lưới
            # Convert set thành list để có thể serialize
            graph_data = {
                artist: list(collaborators)
                for artist, collaborators in network.items()
            }
            
            with open('graph_data.json', 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
                
            print("\nĐã lưu dữ liệu vào channel_data.json và graph_data.json")
            
        except Exception as e:
            print(f"\nLỗi khi lưu dữ liệu: {str(e)}")

    def visualize_network(self) -> None:
        """Trực quan hóa mạng lưới bằng matplotlib"""
        try:
            if len(self.G) == 0:
                print("\nKhông có dữ liệu để trực quan hóa!")
                return
                
            plt.figure(figsize=(20, 20))
            
            # Tính toán layout
            pos = nx.spring_layout(self.G, k=1, iterations=50)
            
            # Tính betweenness centrality
            bc = nx.betweenness_centrality(self.G)
            
            # Vẽ nodes với size dựa trên degree và color dựa trên betweenness
            node_sizes = [self.G.degree(node) * 100 for node in self.G.nodes()]
            nodes = nx.draw_networkx_nodes(self.G, pos, 
                                         node_size=node_sizes,
                                         node_color=list(bc.values()),
                                         cmap=plt.cm.viridis)
            
            # Vẽ edges
            nx.draw_networkx_edges(self.G, pos, alpha=0.2)
            
            # Thêm labels cho nodes có degree cao
            labels = {}
            for node in self.G.nodes():
                if self.G.degree(node) > np.mean([self.G.degree(n) for n in self.G.nodes()]):
                    labels[node] = self.channel_info[node]['name']
            nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
            
            plt.title("Mạng lưới Collab Nghệ sĩ VPop", fontsize=16, pad=20)
            plt.colorbar(nodes, label='Betweenness Centrality')
            plt.axis('off')
            
            # Lưu hình với DPI cao
            plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
            print("\nĐã lưu trực quan hóa vào network_visualization.png")
            
        except Exception as e:
            print(f"\nLỗi khi trực quan hóa: {str(e)}")

    def analyze_metrics(self) -> None:
        """Phân tích các metrics ca mạng lưới"""
        try:
            # 1. Tính toán các metrics cơ bản
            n_nodes = self.G.number_of_nodes()
            n_edges = self.G.number_of_edges()
            density = nx.density(self.G)
            
            # 2. Tính degree distribution
            degrees = [d for n, d in self.G.degree()]
            avg_degree = sum(degrees) / len(degrees)
            
            # 3. Tính centrality
            betweenness = nx.betweenness_centrality(self.G)
            closeness = nx.closeness_centrality(self.G)
            eigenvector = nx.eigenvector_centrality(self.G, max_iter=1000)
            
            # 4. Tìm các nghệ sĩ có ảnh hưởng nhất
            top_degree = sorted(
                [(node, deg) for node, deg in self.G.degree()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            top_betweenness = sorted(
                betweenness.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # 5. Lưu kết quả phân tích
            analysis_results = {
                'basic_metrics': {
                    'nodes': n_nodes,
                    'edges': n_edges,
                    'density': density,
                    'average_degree': avg_degree
                },
                'top_artists': {
                    'by_degree': [
                        {
                            'artist': self.channel_info[node]['artist'],
                            'degree': deg
                        }
                        for node, deg in top_degree
                    ],
                    'by_betweenness': [
                        {
                            'artist': self.channel_info[node]['artist'],
                            'betweenness': score
                        }
                        for node, score in top_betweenness
                    ]
                },
                'centrality_metrics': {
                    node: {
                        'artist': self.channel_info[node]['artist'],
                        'betweenness': betweenness[node],
                        'closeness': closeness[node],
                        'eigenvector': eigenvector[node]
                    }
                    for node in self.G.nodes()
                }
            }
            
            with open('network_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2)
            
            print("\nĐã lưu kết quả phân tích vào network_analysis.json")
            
            # 6. In một số thống kê cơ bản
            print(f"\nThống kê mạng lưới:")
            print(f"- Số lượng nghệ sĩ: {n_nodes}")
            print(f"- Số lượng collab: {n_edges}")
            print(f"- Mật độ mạng lưới: {density:.4f}")
            print(f"- Trung bình số collab: {avg_degree:.2f}")
            
            print("\nTop 5 nghệ sĩ có nhiều collab nhất:")
            for node, deg in top_degree[:5]:
                print(f"- {self.channel_info[node]['artist']}: {deg} collab")
            
        except Exception as e:
            print(f"\nLỗi khi phân tích metrics: {str(e)}")

    def _verify_channel_content(self, channel_url: str, artist_name: str) -> bool:
        """Xác minh nội dung kênh có phải của nghệ sĩ không"""
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                channel_info = ydl.extract_info(channel_url, download=False)
                
                if not channel_info:
                    return False
                    
                # 1. Kiểm tra subscriber count
                subscriber_count = channel_info.get('subscriber_count', 0)
                if subscriber_count < 1000:  # Kênh quá nhỏ
                    return False
                    
                # 2. Kiểm tra video count
                video_count = channel_info.get('video_count', 0) 
                if video_count < 5:  # Quá ít video
                    return False
                    
                # 3. Kiểm tra description
                description = channel_info.get('description', '').lower()
                music_keywords = {
                    'ca sĩ', 'nghệ sĩ', 'rapper', 'producer', 'nhạc sĩ',
                    'musician', 'artist', 'singer', 'composer', 'official',
                    'chính thức', 'channel', 'music', 'entertainment'
                }
                if not any(keyword in description for keyword in music_keywords):
                    return False
                    
                # 4. Kiểm tra tên kênh một lần nữa
                channel_name = channel_info.get('channel', '').lower()
                artist_name_lower = artist_name.lower()
                
                if artist_name_lower not in channel_name:
                    return False
                    
                return True
                
        except Exception:
            return False

def main():
    analyzer = YouTubeNetworkAnalyzer()
    
    # Phân tích mạng lưới và lưu kết quả
    network = analyzer.analyze_network()
    
    # Lưu dữ liệu với network đã phân tích
    analyzer.save_data(network)
    
    # Phân tích metrics
    analyzer.analyze_metrics()
    
    # Trực quan hóa
    analyzer.visualize_network()

if __name__ == "__main__":
    main()