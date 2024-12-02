from yt_dlp import YoutubeDL

video_url = "https://www.youtube.com/watch?v=3xXgMMT813c"
ydl = YoutubeDL()

try:
    info = ydl.extract_info(video_url, download=False)
    print("Title:", info['title'])
    print("Views:", info['view_count'])
    print("Duration (s):", info['duration'])
    print("Description:", info['description'])
    print("Author:", info['uploader'])
except Exception as e:
    print("Lỗi:", str(e))

