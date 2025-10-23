import ast
import re
import matplotlib.pyplot as plt
import pandas as pd

# Reading in prediction results
df = pd.read_csv("Prediction_1/output.csv")

# Converting to lists
def to_list(x):
    # If it's already a list-like string (starts with [), use literal_eval
    if x.startswith("[") and x.endswith("]"):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return [str(v).strip() for v in val]
            else:
                return [str(val)]
        except Exception:
            pass
    # Otherwise, split by commas after removing quotes/brackets
    x = re.sub(r"[\[\]']", "", x)
    parts = [p.strip() for p in x.split(",") if p.strip()]
    return parts

df["Prediction"] = df["Prediction"].apply(to_list)
df["Actual"] = df["Actual"].apply(to_list)

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

df["Category"] = df.apply(lambda row: classify(row["Prediction"], row["Actual"]), axis = 1)

# Counting the instances within each prediction category
category_counts = df["Category"].value_counts().reindex(
    ["Correct", "Partially Correct", "Wrong"], fill_value = 0
)

# Plotting histogram to display the results of Prediction_1 model
plt.figure(figsize=(8,6))
plt.bar(
    category_counts.index,
    category_counts.values,
    # Green = Correct, Yellow = Partially Correct, Red = Wrong
    color=["green", "yellow", "red"],
    edgecolor="black"
)

plt.title("Prediction Accuracy", fontsize = 16)
plt.xlabel("Prediction Category", fontsize = 12)
plt.ylabel("Instances", fontsize = 12)
plt.show()
