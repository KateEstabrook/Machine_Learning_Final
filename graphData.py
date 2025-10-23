import ast
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import PCA

# Read the file
data = []
with open("Prediction_1/Predictions.txt", "r", encoding="utf-8") as f:
    for line in f:
        # Split only the first two commas, to avoid splitting genre lists
        parts = line.strip().split("',", 2)
        print(parts)
        if len(parts) == 3:
            name = parts[0].strip().strip("'").strip('"')
            pred_genres = ast.literal_eval(parts[1].strip().lstrip(',').strip())
            orig_genres = ast.literal_eval(parts[2].strip().lstrip(',').strip())
            data.append((name, pred_genres, orig_genres))

df = pd.DataFrame(data, columns=["game", "predicted", "original"])
print(df)