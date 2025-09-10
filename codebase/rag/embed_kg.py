import pandas as pd
from collections import defaultdict
import pickle
from tqdm import tqdm
from ..utils.bgem3 import batch_encode

df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/datasets/dataverse_files/kg.csv", low_memory=False)
df = df[["relation", "x_index", "y_index"]]

adj_list = defaultdict(list)
for u, v, r in tqdm(zip(df["x_index"].values, df["y_index"].values, df["relation"].values), total=len(df)):
    adj_list[int(u)].append((int(v), str(r)))

out_path = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/kg_adjacency.pkl"
with open(out_path, "wb") as f:
    pickle.dump(adj_list, f, protocol=pickle.HIGHEST_PROTOCOL)

df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/datasets/dataverse_files/disease_features.csv", low_memory=False)

df = df.sort_values("node_index").reset_index(drop=True)

df["Diseases"] = (
    "[disease name]" + df["mondo_name"].fillna("") + " " +
    "[definition]" + df["mondo_definition"].combine_first(df["orphanet_definition"]).fillna("") + " " +
    "[description]" + df["umls_description"].fillna("")
)

df["embed"] = list(batch_encode(df["Diseases"].tolist(), batch_size=64, max_length=8192).cpu().numpy())
df = df[["node_index", "mondo_name", "Diseases", "embed"]]

out_path = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/disease_features_cleaned.pkl"
with open(out_path, "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
