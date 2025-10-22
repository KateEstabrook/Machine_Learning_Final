import ast
import pandas as pd
import os

df = pd.read_csv('games.csv')
df['genres'] = df['genres'].apply(lambda s: ast.literal_eval(s) if pd.notna(s) else [])

genres_ = sorted({g for genre_list in df['genres'] for g in genre_list})
print(genres_)


# for name in genres_:
#     folder_name = name
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#         print(f"Folder '{folder_name}' created.")
#     else:
#         print(f"Folder '{folder_name}' already exists.")
