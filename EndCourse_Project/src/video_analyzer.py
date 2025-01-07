from typing import Set, List, Dict, Tuple
from yt_dlp import YoutubeDL
import time
import pandas as pd
import re

class VideoAnalyzer:
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'force_generic_extractor': True
        }
        # Load danh sách kênh chính thức
        self.official_channels = self._load_official_channels()

    def _load_official_channels(self) -> pd.DataFrame:
        """Load danh sách kênh chính thức từ CSV"""
        try:
            return pd.read_csv('output/official_channels.csv')
        except Exception as e:
            print(f"Lỗi khi load danh sách kênh: {str(e)}")
            return pd.DataFrame()

    def get_channel_videos(self, channel_url: str, min_videos: int = 30) -> List[str]:
        """Lấy danh sách video chính thức của kênh"""
        videos = []
        
        try:
            # Cấu hình yt-dlp để bỏ qua signature extraction
            ydl_opts = {
                'quiet': True,
                'extract_flat': True,
                'force_generic_extractor': True,
                'no_warnings': True,
                'ignoreerrors': True
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                # Lấy tên nghệ sĩ
                artist_name = self.official_channels[
                    self.official_channels['channel_url'] == channel_url
                ]['artist_name'].iloc[0]

                # Tìm kiếm trực tiếp từ kênh
                try:
                    # Thử lấy uploads playlist của kênh
                    channel_id = channel_url.split('/')[-1]
                    if '@' in channel_id:
                        channel_info = ydl.extract_info(channel_url, download=False)
                        channel_id = channel_info.get('id', '')
                    
                    if channel_id:
                        playlist_url = f'https://www.youtube.com/channel/{channel_id}/videos'
                        results = ydl.extract_info(playlist_url, download=False)
                        
                        if results and 'entries' in results:
                            for entry in results['entries']:
                                if not entry:
                                    continue
                                    
                                title = entry.get('title', '').lower()
                                video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                                
                                if self._is_music_video(title):
                                    videos.append(video_url)
                                    print(f"Tìm thấy MV: {entry.get('title', '')}")
                                    
                                if len(videos) >= min_videos:
                                    break
                except:
                    pass

                # Nếu chưa đủ video, thử tìm theo tên nghệ sĩ
                if len(videos) < min_videos:
                    search_query = f"ytsearch50:\"{artist_name}\" official music video"
                    try:
                        results = ydl.extract_info(search_query, download=False)
                        if results and 'entries' in results:
                            for entry in results['entries']:
                                if not entry:
                                    continue
                                    
                                uploader_url = entry.get('uploader_url', '') or entry.get('channel_url', '')
                                if uploader_url and uploader_url.lower() == channel_url.lower():
                                    title = entry.get('title', '').lower()
                                    video_id = entry.get('id', '')
                                    
                                    if video_id and self._is_music_video(title):
                                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                                        if video_url not in videos:
                                            videos.append(video_url)
                                            print(f"Tìm thấy MV: {entry.get('title', '')}")
                                            
                                if len(videos) >= min_videos:
                                    break
                                    
                    except Exception as e:
                        print(f"Lỗi khi tìm video: {str(e)}")

        except Exception as e:
            print(f"Lỗi khi lấy video từ kênh {channel_url}: {str(e)}")
            
        return videos

    def _is_music_video(self, title: str, description: str = '') -> bool:
        """Kiểm tra xem video có phải là MV chính thức không dùng regex"""
        title = title.lower()
        description = description.lower()

        # Pattern cho MV chính thức
        official_patterns = [
            r'official\s*(music)?\s*video',
            r'official\s*m/?v',
            r'\[official\].*\b(mv|m/v)\b',
            r'\b(mv|m/v)\b.*\[official\]',
            r'official\s*(lyric|lyrics)\s*video',
            r'official\s*performance\s*video',
            r'official\s*visualizer',
            r'official\s*audio',
            r'\[mv\s*lyrics\]',
            r'\[performance\]'
        ]

        # Pattern loại trừ
        exclude_patterns = [
            r'\b(behind\s*the\s*scenes|making|teaser|trailer)\b',
            r'\b(reaction|review|react|cover)\b',
            r'\b(tiktok|shorts|clip)\b',
            r'\b(karaoke|remix|lofi)\b',
            r'\b(talkshow|interview|gameshow|vlog)\b',
            r'đi\s*quay|chia\s*sẻ|tâm\s*sự|trải\s*lòng',
            r'\b(bts|practice|rehearsal)\b'
        ]

        # Kiểm tra trong title và description
        text_to_check = f"{title}\n{description}"
        
        # Kiểm tra các pattern chính thức
        is_official = any(re.search(pattern, text_to_check, re.I) for pattern in official_patterns)
        
        # Kiểm tra các pattern loại trừ
        is_excluded = any(re.search(pattern, text_to_check, re.I) for pattern in exclude_patterns)
        
        # Thêm điều kiện về độ dài title (tránh video quá ngắn/dài)
        title_words = len(title.split())
        if title_words < 3 or title_words > 20:
            return False

        return is_official and not is_excluded

    def get_video_collaborators(self, video_url: str, channel_artist: str) -> Set[str]:
        """Lấy danh sách nghệ sĩ collab trong video sử dụng regex thông minh"""
        collaborators = set()
        
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                video_info = ydl.extract_info(video_url, download=False)
                if not video_info:
                    return collaborators

                title = video_info.get('title', '').lower()
                description = video_info.get('description', '').lower()

                # Tìm tất cả nghệ sĩ trong danh sách official
                known_artists = {name.lower(): name for name in self.official_channels['artist_name']}

                def extract_artists_from_text(text: str) -> Set[str]:
                    """Trích xuất nghệ sĩ từ text với regex patterns mạnh mẽ"""
                    found = set()
                    text = text.lower()

                    # Regex patterns mạnh mẽ cho các format collab phổ biến
                    collab_patterns = [
                        # Format: Nghệ sĩ ft/feat/with nghệ sĩ khác
                        r'([^-|\n]+?)\s*(?:ft\.?|feat\.?|featuring|với|with|cùng|và|&)\s+([^-|\n]+?)(?=\s*[-|\(\)\[\]]|\s*$)',
                        
                        # Format: Nghệ sĩ x/X nghệ sĩ khác
                        r'([^-|\n]+?)\s*(?:x|×|X)\s+([^-|\n]+?)(?=\s*[-|\(\)\[\]]|\s*$)',
                        
                        # Format: Tên bài - Nghệ sĩ ft/x nghệ sĩ khác
                        r'[-|]\s*([^-|\n]+?)\s*(?:ft\.?|feat\.?|featuring|với|with|x|×|X|&|và)\s+([^-|\n]+?)(?=\s*[-|\(\)\[\]]|\s*$)',
                        
                        # Format: Nghệ sĩ trong ngoặc
                        r'[\(\[]\s*(?:ft\.?|feat\.?|featuring|với|with|cùng)\s*:?\s*([^\)\]]+)[\)\]]',
                        
                        # Format: Credits trong description
                        r'(?:ca\s*sĩ|trình\s*bày|thể\s*hiện|vocal(?:ist)?s?|singer|voice|performed\s+by)[:\s]+([^-|\n\(\)\[\]]+)',
                        
                        # Format: Danh sách nghệ sĩ phân cách bằng dấu phẩy
                        r'(?:artists?|nghệ\s*sĩ|performer|thực\s*hiện|biểu\s*diễn)[:\s]+([^-|\n\(\)\[\]]+)',
                        
                        # Format: Tên bài | Nghệ sĩ x Nghệ sĩ
                        r'\|\s*([^|\n]+?)\s*(?:x|×|X)\s*([^|\n]+?)(?:\s*\||\s*$)',
                        
                        # Format: Nghệ sĩ ở đầu hoặc sau dấu gạch ngang
                        r'^([^-|\n]+?)(?=\s*[-|\(\)\[\]])',
                        r'[-|]\s*([^-|\n\(\)\[\]]+?)(?=\s*$|\s*\(|\s*\[)',
                        
                        # Format: Collab trong credits
                        r'(?m)^[-•*+]\s*([^:\n]+?)(?::\s*|\s+-\s*)([^\n]+)$',  # Bullet points
                        r'(?m)^(?:performed|song|music|vocals?)\s+by[:\s]+([^\n]+)$',  # Credits line
                        
                        # Format: Produced/Mixed/Mastered credits
                        r'(?:produced|mixed|mastered|sản\s*xuất|phối\s*khí|hoà\s*âm)\s+by[:\s]+([^\n]+)',
                        
                        # Format: Duet/Song with
                        r'(?:duet|song|hát\s*cùng|song\s*ca)\s+(?:with|cùng|với|by|bởi)[:\s]+([^\n]+)',
                        
                        # Format: Starring/Featuring artists
                        r'(?:starring|featuring|diễn\s*viên|vai\s*diễn)[:\s]+([^\n]+)',
                        
                        # Format: Music credits
                        r'(?:music|beat|instrumental|nhạc)\s+(?:by|từ|của)[:\s]+([^\n]+)',
                        
                        # Format: Arrangement/Composition credits
                        r'(?:arranged|composed|written|sáng\s*tác|viết\s*lời)\s+by[:\s]+([^\n]+)',
                        
                        # Format: Collaboration credits
                        r'(?:collab(?:oration)?|hợp\s*tác|kết\s*hợp)\s+(?:with|cùng|và)[:\s]+([^\n]+)',
                        
                        # Format: Special appearance
                        r'(?:special|guest|khách\s*mời)\s+(?:appearance|featuring|performer)[:\s]+([^\n]+)'
                    ]

                    # Xử lý từng pattern
                    for pattern in collab_patterns:
                        matches = re.finditer(pattern, text, re.I | re.UNICODE)
                        for match in matches:
                            # Xử lý tất cả các group trong match
                            for group_idx in range(1, len(match.groups()) + 1):
                                if not match.group(group_idx):
                                    continue
                                    
                                # Tách các nghệ sĩ theo dấu phân cách
                                artists = re.split(r'\s*[,&x×X]\s*|\s+(?:và|with|cùng|featuring|feat\.?|ft\.?)\s+', match.group(group_idx))
                                
                                for artist in artists:
                                    artist = artist.strip('.,()[]{}"\' \n\t-|')
                                    if not artist:
                                        continue
                                        
                                    artist_lower = artist.lower()
                                    if artist_lower in known_artists and artist_lower != channel_artist.lower():
                                        # Kiểm tra xem có phải là một phần của tên dài hơn không
                                        is_part_of_longer = any(
                                            other != artist_lower and artist_lower in other and 
                                            re.search(r'\b' + re.escape(other) + r'\b', text, re.I)
                                            for other in known_artists
                                        )
                                        
                                        if not is_part_of_longer:
                                            found.add(known_artists[artist_lower])

                    return found

                # Xử lý title
                collaborators.update(extract_artists_from_text(title))

                # Xử lý description theo từng dòng
                for line in description.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    collaborators.update(extract_artists_from_text(line))

        except Exception as e:
            print(f"\nLỗi khi lấy collaborators từ {video_url}: {str(e)}")
            
        return collaborators

    def _get_exact_artist_name(self, artist_lower: str) -> str:
        """Lấy tên chính xác của nghệ sĩ từ tên viết thường"""
        try:
            return self.official_channels[
                self.official_channels['artist_name'].str.lower() == artist_lower
            ]['artist_name'].iloc[0]
        except:
            return None

    def _is_known_artist(self, artist_name: str) -> bool:
        """Kiểm tra xem nghệ sĩ có trong danh sách official không"""
        artist_name = artist_name.lower().strip()
        known_artists = set(self.official_channels['artist_name'].str.lower())
        return artist_name in known_artists 

    # Xử lý đặc biệt cho nghệ sĩ xuất hiện ở đầu title
    def extract_leading_artist(self, text: str) -> Set[str]:
        """Trích xuất nghệ sĩ xuất hiện ở đầu text"""
        found = set()
        
        # Tách phần đầu tiên của text (trước dấu gạch ngang hoặc các ký tự đặc biệt)
        first_part = re.split(r'[-|\(\)\[\]]', text)[0].strip()
        
        # Tách theo khoảng trắng để lấy từ đầu tiên hoặc cụm từ
        words = first_part.split()
        if not words:
            return found
            
        # Thử các combination của các từ đầu tiên
        for i in range(len(words), 0, -1):
            potential_artist = ' '.join(words[:i]).lower()
            if potential_artist in known_artists and potential_artist != channel_artist.lower():
                # Kiểm tra xem có phải là một phần của tên dài hơn không
                is_part_of_longer = any(
                    other != potential_artist and potential_artist in other and 
                    re.search(r'\b' + re.escape(other) + r'\b', text, re.I)
                    for other in known_artists
                )
                
                if not is_part_of_longer:
                    found.add(known_artists[potential_artist])
                    break  # Dừng khi tìm thấy nghệ sĩ dài nhất match
        
        return found 

    def _remove_accents(self, text: str) -> str:
        """Bỏ dấu tiếng Việt"""
        s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
        s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
        return re.sub(f'[{s1}]', lambda x: s0[s1.index(x.group(0))], text) 