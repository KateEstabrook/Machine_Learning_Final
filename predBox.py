import ast
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reading in prediction results
df = pd.read_csv("Prediction_1/output.csv")

# Converting string values to Python lists
def to_list(x):
    # If it's already a list-like string (starts with [), use literal_eval
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            val = ast.literal_eval(x)
            return list(val) if isinstance(val, list) else [val]
        except Exception:
            pass
    # Otherwise, split by commas after removing quotes/brackets
    if isinstance(x, str):
        x = re.sub(r"[\[\]']", "", x)
        return [p.strip() for p in x.split(",") if p.strip()]
    return []

df["Prediction"] = df["Prediction"].apply(to_list)
df["Actual"] = df["Actual"].apply(to_list)
df["Confidence"] = df["Confidence"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Classifying each row as either Wrong, Partially Correct, or Correct
def classify(pred, actual):
    pred_set, act_set = set(pred), set(actual)
    overlap = len(pred_set & act_set)
    if overlap == 0:
        return "Wrong"
    elif pred_set == act_set:
        return "Correct"
    else:
        return "Partially Correct"

df["Category"] = df.apply(lambda r: classify(r["Prediction"], r["Actual"]), axis=1)
# Computing the average confidence score for each game
df["avg_conf"] = df["Confidence"].apply(
    lambda lst: float(np.mean(lst)) if lst else np.nan
)

# Drop rows with no confidence values
plot_df = df.dropna(subset=["avg_conf"])

# Plotting box plot showing the distribution of confidence values
plt.figure(figsize=(8,6))
box = plt.boxplot(
    # Confidence values grouped by prediction type
    [plot_df.loc[plot_df["Category"] == cat, "avg_conf"]
     for cat in ["Correct", "Partially Correct", "Wrong"]],
    labels=["Correct", "Partially Correct", "Wrong"],
    patch_artist = True
)

# Coloring each box (Green = Correct, Yellow = Partially Correct, Red = Wrong)
colors = ["green", "yellow", "red"]
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel("Average Confidence")
plt.title("Average Prediction Confidence by Category")
plt.tight_layout()
plt.savefig("graphs/boxplot.png")
plt.show()