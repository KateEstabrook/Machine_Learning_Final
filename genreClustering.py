import ast
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

df = pd.read_csv("games.csv")

# Converting genre strings into actual Python lists
def parse_genres(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception:
        return []

df["genres_list"] = df["genres"].apply(parse_genres)

# Keeping rows that have at least one genre
df = df[df["genres_list"].apply(len) > 0]

# Combining text columns as seen in Prediction_2 
text_cols = ["name", "short_description", "about_the_game", "detailed_description"]
df["combined_text"] = df[text_cols].fillna("").agg(" ".join, axis=1)

# TF-IDF vectorization of game descriptions
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(df["combined_text"])

# Finding unique genres
all_genres = [
    genre
    for sublist in df['genres_list']
    if isinstance(sublist, list)
    for genre in sublist
]
unique_genres = sorted(set(all_genres))

genre_vecs = []  
genre_names = [] 

for g in unique_genres:
    # Does a game contain genre g
    has_genre = df["genres_list"].apply(lambda lst: g in lst).to_numpy()
    # Keeping only rows that have genre g
    X_g = X[has_genre]
    # Averaging the TF-IDF vector over games with genre g
    avg = np.asarray(X_g.mean(axis=0)).ravel()
    genre_vecs.append(avg)
    genre_names.append(g)

# Turn list of vectors into a matrix 
genre_vecs = np.vstack(genre_vecs)

# Running K-Means to cluster genres based on average TF-IDF
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_ids = kmeans.fit_predict(genre_vecs)

# Using PCA to reduce vector size to 2 dimensions for plotting
coords_2d = PCA(n_components=2, random_state=42).fit_transform(genre_vecs)

# Plotting genre clusters
plt.figure(figsize=(10, 8))

# Color map with colors for each genre
cmap = plt.cm.get_cmap("tab20", len(genre_names))

# Looping through each genre
for i in range(len(genre_names)):
    # Coordinates of genre in PCA space
    x, y = coords_2d[i]
    # Name of genre
    genre = genre_names[i]
    # Cluster for genre
    c = cluster_ids[i]    

    # Plotting genre as point
    plt.scatter(
        x, y,
        color=cmap(c),   # Color based on cluster   
        s=90,
        label=genre        
    )

# Plotting genre legend
plt.legend()
plt.title("K-Means Genre Clusters")
plt.tight_layout()
plt.savefig("graphs/genre_clusters.png")
plt.show()