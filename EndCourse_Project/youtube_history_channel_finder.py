from yt_dlp import YoutubeDL
import re
import time
from datetime import datetime
from collections import defaultdict

class HistoryChannelFinder:
    def __init__(self):
        # Regex patterns cho việc nhận diện nội dung lịch sử
        self.history_patterns = [
            r'(?i)(lịch\s*sử|history|historical)',
            r'(?i)(triều\s*đại|dynasty|feudal)',
            r'(?i)(cổ\s*đại|ancient|archaeology)',
            r'(?i)(văn\s*minh|civilization)',
            r'(?i)(chiến\s*tranh|war|military)',
            r'(?i)(khảo\s*cổ|di\s*tích|di\s*sản)',
            r'(?i)(thời\s*kỳ|period|era)',
            r'(?i)(đế\s*chế|empire|imperial)',
            r'(?i)(phong\s*kiến|medieval)',
            r'(?i)(khởi\s*nghĩa|rebellion|revolution)'
        ]
        
        # Compile regex patterns để tối ưu performance
        self.compiled_patterns = [re.compile(pattern) for pattern in self.history_patterns]
        
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True,
            'skip_download': True,
            'format': 'best',
            'max_downloads': 50,  # Giới hạn số lượng video phân tích
            'sleep_interval': 1,  # Delay giữa các request
            'max_sleep_interval': 5,
        }

    def search_history_channels(self, max_results=100):
        """Tìm kiếm các kênh lịch sử thông qua search"""
        search_queries = [
            'kênh lịch sử việt nam',
            'history channel vietnam',
            'historical documentary channel',
            'kênh tài liệu lịch sử',
            'vietnamese history channel'
        ]
        
        found_channels = set()
        channels_data = []

        for query in search_queries:
            try:
                with YoutubeDL(self.ydl_opts) as ydl:
                    # Tìm kiếm với từ khóa
                    search_url = f"ytsearch{max_results}:{query}"
                    results = ydl.extract_info(search_url, download=False)
                    
                    if not results.get('entries'):
                        continue

                    for entry in results['entries']:
                        channel_id = entry.get('channel_id')
                        if channel_id and channel_id not in found_channels:
                            found_channels.add(channel_id)
                            channel_data = self.analyze_channel(f"https://www.youtube.com/channel/{channel_id}")
                            if channel_data and channel_data['history_score'] > 0.6:
                                channels_data.append(channel_data)
                            time.sleep(1)  # Delay giữa các request

            except Exception as e:
                print(f"Lỗi khi tìm kiếm với từ khóa '{query}': {str(e)}")
                continue

        return channels_data

    def calculate_history_score(self, text):
        """Tính điểm liên quan đến lịch sử của một đoạn text"""
        if not isinstance(text, str) or not text.strip():
            return (0, {})  # Luôn trả về tuple
        
        text = text.lower()
        score = 0
        matches = defaultdict(int)
        
        try:
            for pattern in self.compiled_patterns:
                found_matches = pattern.findall(text)
                if found_matches:
                    matches[pattern.pattern] += len(found_matches)
                    score += len(found_matches)
            
            # Chuẩn hóa điểm về thang 0-1
            max_possible_score = len(self.compiled_patterns) * 2  # Giả sử mỗi pattern có thể match 2 lần
            normalized_score = min(1.0, score / max_possible_score)
            
            return (normalized_score, dict(matches))  # Đảm bảo luôn trả về tuple
            
        except Exception as e:
            print(f"Lỗi khi tính điểm cho text: {str(e)}")
            return (0, {})  # Trả về tuple mặc định nếu có lỗi

    def analyze_channel(self, channel_url):
        """Phân tích chi tiết một kênh YouTube"""
        try:
            with YoutubeDL(self.ydl_opts) as ydl:
                channel_info = ydl.extract_info(channel_url, download=False)
                
                if not channel_info or not channel_info.get('entries'):
                    print(f"Không tìm thấy video trong kênh {channel_url}")
                    return None

                total_score = 0
                keyword_stats = defaultdict(int)
                video_analysis = []
                
                # Phân tích từng video
                for video in channel_info['entries']:
                    try:
                        title = video.get('title', '')
                        description = video.get('description', '')
                        
                        # Đảm bảo nhận được tuple với 2 phần tử
                        title_result = self.calculate_history_score(title)
                        desc_result = self.calculate_history_score(description)
                        
                        # Kiểm tra kết quả trả về
                        if not isinstance(title_result, tuple) or not isinstance(desc_result, tuple):
                            print(f"Lỗi: Kết quả không phải tuple - title_result: {type(title_result)}, desc_result: {type(desc_result)}")
                            continue
                            
                        title_score, title_matches = title_result
                        desc_score, desc_matches = desc_result
                        
                        # Cập nhật thống kê từ khóa
                        for pattern, count in {**title_matches, **desc_matches}.items():
                            keyword_stats[pattern] += count
                        
                        video_score = (title_score * 2 + desc_score) / 3
                        total_score += video_score
                        
                        video_analysis.append({
                            'title': title,
                            'url': video.get('webpage_url'),
                            'view_count': video.get('view_count', 0),
                            'history_score': video_score,
                            'upload_date': video.get('upload_date', '')
                        })
                        
                    except Exception as e:
                        print(f"Lỗi khi phân tích video trong kênh {channel_url}: {str(e)}")
                        continue

                if not video_analysis:
                    print(f"Không có video nào được phân tích thành công trong kênh {channel_url}")
                    return None
                    
                avg_score = total_score / len(video_analysis)
                
                return {
                    'channel_name': channel_info.get('uploader', ''),
                    'channel_url': channel_url,
                    'subscriber_count': channel_info.get('subscriber_count', 0),
                    'total_videos': len(video_analysis),
                    'history_score': avg_score,
                    'keyword_stats': dict(keyword_stats),
                    'video_samples': sorted(video_analysis, 
                                         key=lambda x: x['history_score'], 
                                         reverse=True)[:5],
                    'analysis_date': datetime.now().isoformat()
                }

        except Exception as e:
            print(f"Lỗi khi phân tích kênh {channel_url}: {str(e)}")
            return None

    def print_detailed_results(self, channels):
        """In kết quả phân tích chi tiết"""
        print(f"\nĐã tìm thấy {len(channels)} kênh YouTube về lịch sử:")
        
        for idx, channel in enumerate(sorted(channels, 
                                          key=lambda x: x['history_score'], 
                                          reverse=True), 1):
            print(f"\n{idx}. {channel['channel_name']}")
            print(f"   URL: {channel['channel_url']}")
            print(f"   Điểm lịch sử: {channel['history_score']:.2f}")
            print(f"   Số người đăng ký: {channel['subscriber_count']:,}")
            print(f"   Tổng số video: {channel['total_videos']}")
            
            print("\n   Từ khóa phổ biến:")
            for pattern, count in sorted(channel['keyword_stats'].items(), 
                                      key=lambda x: x[1], 
                                      reverse=True)[:5]:
                print(f"      - {pattern}: {count} lần")
            
            print("\n   Video tiêu biểu:")
            for video in channel['video_samples'][:3]:
                print(f"      - {video['title']}")
                print(f"        Lượt xem: {video['view_count']:,}")
                print(f"        Điểm lịch sử: {video['history_score']:.2f}")

# Sử dụng
finder = HistoryChannelFinder()
channels = finder.search_history_channels()
finder.print_detailed_results(channels)

