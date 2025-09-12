# import pandas as pd
# import pickle
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# with open("D:/Lab/Research/EMERGE-REPLICATE/rag/curated_data/disease_features_cleaned.pkl", "rb") as f:
#     df = pickle.load(f)

# with open("D:/Lab/Research/EMERGE-REPLICATE/rag/curated_data/kg_adjacency.pkl", "rb") as f:
#     adj = pickle.load(f)

# corpus_embs = torch.stack([torch.tensor(e) for e in df["embed"].values]).to(device)

# def match(query, corpus_embs, df, threshold=0.6):
    
#     # idx_sorted = cosine_filter(query_emb, corpus_embs, threshold=threshold)
#     return 
#     matches = df.iloc[idx_sorted]
#     # print(matches.head())
#     nodes = set(matches["node_index"].tolist())
#     return matches

# if __name__ == "__main__":
#     query = "fluid in the lungs causing breathing difficulty"
#     matches = match(query, corpus_embs, df, threshold=0.6)
#     # print(matches[["mondo_name", "mondo_definition"]])