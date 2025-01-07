#xử lý trùng lặp nghệ sĩ 
import pandas as pd

df = pd.read_csv('Vietnamese Artists Unique.csv')
df_unique = df.drop_duplicates(subset=['Artist'])
df_unique.to_csv('Vietnamese Artists Unique.csv', index=False)