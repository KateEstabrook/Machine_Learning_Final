import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

# Makes sure data is parsed in correctly for use
def safe_list(x):
    try:
        return ast.literal_eval(x) if isinstance(x, str) else []
    except:
        return []

# Create and load dataframes
df = pd.read_csv("Prediction_1/output.csv")

df['Prediction'] = df['Prediction'].apply(safe_list)
df['Actual']     = df['Actual'].apply(safe_list)
df['Confidence'] = df['Confidence'].apply(safe_list)

# Multi-label binarizations
mlb = MultiLabelBinarizer()
y_pred = mlb.fit_transform(df['Prediction'])
y_true = mlb.transform(df['Actual'])  # use same classes as prediction

genres = mlb.classes_
n = len(genres)

# Create confusion matrix  
# Rows = Actual genre
# Cols = Predicted genre
conf_matrix = np.zeros((n, n), dtype=int)

for i in range(len(df)):
    for a_idx, actual_label in enumerate(y_true[i]):
        if actual_label == 1:
            # For each actual genre, check which predicted genres were selected
            for p_idx, pred_label in enumerate(y_pred[i]):
                if pred_label == 1:
                    conf_matrix[a_idx][p_idx] += 1

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(conf_matrix, cmap="viridis")

# Show numbers on the heatmap
for i in range(n):
    for j in range(n):
        ax.text(
            j, i, str(conf_matrix[i, j]),
            ha="center", va="center",
            color="white" if conf_matrix[i, j] > np.max(conf_matrix)/2 else "black",
            fontsize=8
        )

# Axis labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(genres, rotation=45, ha="right")
ax.set_yticklabels(genres)

ax.set_xlabel("Predicted Genre")
ax.set_ylabel("Actual Genre")
ax.set_title("Multi-label Genre Confusion Matrix")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("graphs/confusion_matrix.png")
plt.show()