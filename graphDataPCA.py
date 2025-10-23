import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

# Function for safely creating df[Predictions]
# Needed since some games only have 1 genre while others have a list of them
def safe_eval(x):
    try:
        if x.startswith('['):
            return ast.literal_eval(x)
        else:
            return ["No genre"]
    except:
        return ["No genre"]

# Get panadas dataframe from csv
df = pd.read_csv("Prediction_1/output.csv")
print(df.head().to_string(index=False))

# Convert csv strings to actual lists
df['Prediction'] = df['Prediction'].apply(safe_eval)

# Find the dominant genre of the prediction
df['DominantGenre'] = df['Prediction'].apply(lambda x: x[0] if len(x) > 0 else "No genre")

# One-hot encode and combine predictions for PCA 
mlb = MultiLabelBinarizer()
predictions_encoded = mlb.fit_transform(df['Prediction'])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(predictions_encoded)

# Set up colors for scatter spot 
unique_genres = sorted(df['DominantGenre'].unique())
genre_to_int = {g: i for i, g in enumerate(unique_genres)}
colors_idx = df['DominantGenre'].map(genre_to_int).values
cmap = plt.get_cmap('tab20')  # tab10 has 10 distinct colors, use 'tab20' if more needed
num_colors = len(unique_genres)
colors = [cmap(i % 10) for i in range(num_colors)]

# Plot graph
plt.figure(figsize=(10,7))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=[colors[i] for i in colors_idx], alpha=0.8)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label=genre,
                          markerfacecolor=colors[i], markersize=8)
                   for i, genre in enumerate(unique_genres)]
plt.legend(handles=legend_elements, title="Dominant Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("PCA of Game Genres")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.show()