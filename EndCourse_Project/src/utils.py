import pandas as pd
from typing import Set

def load_artists_from_csv() -> Set[str]:
    """Load danh sách nghệ sĩ từ CSV"""
    try:
        df = pd.read_csv('data/Vietnamese_Artists_Unique.csv')
        artists = set(df['Artist'].str.strip().unique())
        print(f"Đã đọc {len(artists)} nghệ sĩ từ CSV")
        return artists
    except Exception as e:
        print(f"Lỗi khi đọc file CSV: {str(e)}")
        return set() 