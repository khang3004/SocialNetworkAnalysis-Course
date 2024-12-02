import facebook
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd
import os

class FacebookCollector:
    def __init__(self, access_token):
        try:
            self.access_token = access_token
            self.graph = facebook.GraphAPI(access_token=self.access_token)
        except Exception as e:
            print(f"Error initializing FacebookCollector: {e}")
            
    def check_token_validity(self):
        try:
            me = self.graph.get_object(id='me', fields='id, name')
            print(f"User ID: {me['id']}, Name: {me['name']}")
            return True
        except facebook.GraphAPIError as e:
            print(f"Error checking token validity: {e}")
            return False
            
    def collect_data(self, user_id, limit=10):
        try:
            fields = (
                'id,'
                'message,'
                'created_time,'
                'comments.limit(100).summary(true){created_time,from{id,name},message,reactions},'
                'reactions.limit(100).summary(true){id,type,name},'
                'shares,'
                'type'
            )
            
            posts = self.graph.get_object(id='me/feed', fields=fields)
            print(f"Collected {len(posts)} posts")
            
            # Lưu dữ liệu thô
            self.save_to_json(posts['data'], 'facebook_posts.json')
            
            # Chuyển đổi và lưu dữ liệu dạng bảng
            self.save_to_excel(posts['data'], 'facebook_posts.xlsx')
            
            return posts['data']
        except Exception as e:
            print(f"Error collecting data: {e}")
            return []

    def save_to_json(self, data, filename):
        """Lưu dữ liệu dưới dạng JSON"""
        try:
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Đã lưu dữ liệu JSON vào {filepath}")
        except Exception as e:
            print(f"Lỗi khi lưu JSON: {e}")

    def save_to_excel(self, data, filename):
        """Chuyển đổi và lưu dữ liệu dưới dạng Excel"""
        try:
            # Chuẩn bị dữ liệu cho DataFrame
            processed_data = []
            for post in data:
                post_data = {
                    'post_id': post.get('id'),
                    'message': post.get('message'),
                    'created_time': post.get('created_time'),
                    'type': post.get('type'),
                    'comments_count': post.get('comments', {}).get('summary', {}).get('total_count', 0),
                    'reactions_count': post.get('reactions', {}).get('summary', {}).get('total_count', 0),
                    'shares_count': post.get('shares', {}).get('count', 0) if post.get('shares') else 0
                }
                processed_data.append(post_data)

            # Tạo DataFrame
            df = pd.DataFrame(processed_data)
            
            # Lưu file
            output_dir = 'output'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            filepath = os.path.join(output_dir, filename)
            df.to_excel(filepath, index=False)
            print(f"Đã lưu dữ liệu Excel vào {filepath}")
        except Exception as e:
            print(f"Lỗi khi lưu Excel: {e}")

    def get_user_posts(self, user_id):
        posts = self.graph.get_object(id=user_id, fields='posts')
        return posts['posts']

def main():
    ACCESS_TOKEN = open("access_token_GraphAPI.txt", "r").read()
    collector = FacebookCollector(ACCESS_TOKEN)

    if collector.check_token_validity():
        collector.collect_data(user_id='me', limit=4)
    else:
        print("Invalid access token")

if __name__ == "__main__":
    main()
