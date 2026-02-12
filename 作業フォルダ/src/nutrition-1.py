# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("utsavdey1410/food-nutrition-dataset")

print("Path to dataset files:", path)

# %%
import pandas as pd
import os
import glob

path = "C:/Users/bi23014/.cache/kagglehub/datasets/utsavdey1410/food-nutrition-dataset/versions/1/FINAL FOOD DATASET"

all_files = glob.glob(os.path.join(path, "*.csv"))

df_list = []

for filename in all_files:
     df = pd.read_csv(filename)
     df_list.append(df)

merged_df = pd.concat(df_list, ignore_index=True)
merged_df.to_csv("merged_file.csv", index=False)  # ファイル名を必要に応じて変更してください

# %%
path = "./merged_file.csv"
df = pd.read_csv(path)

name = df["food"].to_list()
calory = df["Caloric Value"].to_list()
protein = df["Protein"].to_list()
fat = df["Fat"].to_list()
carbohydrates = df["Carbohydrates"].to_list()
fiber = df["Dietary Fiber"].to_list()

# %%
train_data = df.drop(columns=["food"])

# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# KMeans モデルを学習（教師なし学習）
model = KMeans(n_clusters=2)
model.fit(train_data)


# ... (your existing code to load data and train the KMeans model) ...

# Get cluster labels
labels = model.labels_

# Create cross-tabulation table
crosstab = pd.crosstab(df['food'], labels, margins=True)

# Display the table
print(crosstab)


