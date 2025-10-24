import ast
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.manifold import TSNE

# Function for safely creating df[Predictions]
def safe_eval(x):
    try:
        if x.startswith('['):
            return ast.literal_eval(x)
        else:
            return ["No genre"]
    except:
        return ["No genre"]

# Load data
df = pd.read_csv("Prediction_1/output.csv")

# Convert csv strings to lists
df['Prediction'] = df['Prediction'].apply(safe_eval)

# Find dominant genre
df['DominantGenre'] = df['Prediction'].apply(lambda x: x[0] if len(x) > 0 else "No genre")

# One-hot encode predictions
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['Prediction'])

# Standardize for better t-SNE performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(X_scaled)

# Set up colors and markers
unique_genres = sorted(df['DominantGenre'].unique())
num_genres = len(unique_genres)
cmap = plt.get_cmap('hsv')
colors = [cmap(i / num_genres) for i in range(num_genres)]
markers = ['+', 'x']
genre_to_style = {genre: (colors[i], markers[i % len(markers)]) for i, genre in enumerate(unique_genres)}

# Plot t-SNE scatter
fig, ax = plt.subplots(figsize=(8, 7))
for genre in unique_genres:
    mask = df['DominantGenre'] == genre
    color, marker = genre_to_style[genre]
    ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
               c=[color], marker=marker, alpha=0.8, label=genre)

ax.set_title("t-SNE Cloud")

# Create legend with colors and markers
legend_elements = [
    Line2D([0], [0], marker=genre_to_style[genre][1], color='w', label=genre,
           markeredgecolor=genre_to_style[genre][0], markersize=8, linestyle='None')
    for genre in unique_genres
]
ax.legend(handles=legend_elements, title="Dominant Genre", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("graphs/t-SNE_graph.png")
plt.show()
