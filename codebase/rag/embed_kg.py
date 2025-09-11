import pandas as pd
from collections import defaultdict
import pickle
from tqdm import tqdm
from ..utils.bgem3 import batch_encode
import h5py
import numpy as np
import torch
from ..utils.clinical_longformer import langchain_chunk_embed
from typing import List, Optional, Tuple
from FlagEmbedding import BGEM3FlagModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device, trust_remote_code=True)

if __name__ == "__main__":
    # Create adjacency list
    df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/datasets/dataverse_files/kg.csv", low_memory=False)
    df = df[["relation", "x_index", "y_index"]]

    adj_list = defaultdict(list)
    for u, v, r in tqdm(zip(df["x_index"].values, df["y_index"].values, df["relation"].values), total=len(df), desc="Creating adjacency list"):
        adj_list[int(u)].append((int(v), str(r)))

    with open("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/kg_adjacency.pkl", "wb") as f:
        pickle.dump(adj_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Create disease feature embeddings
    df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/datasets/dataverse_files/disease_features.csv", low_memory=False)
    df = df.sort_values("node_index").reset_index(drop=True)

    df["Diseases"] = (
        "[disease name]" + df["mondo_name"].fillna("") + " " +
        "[definition]" + df["mondo_definition"].combine_first(df["orphanet_definition"]).fillna("") + " " +
        "[description]" + df["umls_description"].fillna("")
    )
    df["embed"] = list(batch_encode(df["Diseases"].tolist(), batch_size=64, max_length=8192).cpu().numpy())
    df = df[["node_index", "mondo_name", "Diseases"]]

    with open("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/disease_features_cleaned.pkl", "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Embed notes and save to HDF5
    h5_out = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/notes_embeddings.h5"

    with h5py.File(h5_out, "w") as f:
        pass

    notes_df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/preprocessing-bach/processed/notes.csv")
    for idx, row in tqdm(notes_df.iterrows(), total=len(notes_df), desc="Embedding notes and saving to HDF5"):
        patient_id = row["PatientID"]
        text = row["Text"]
        
        with h5py.File(h5_out, "a") as h5:
            grp = h5.create_group(str(patient_id))
            grp.create_dataset("PatientID", data=np.asarray(patient_id, dtype="int64"))
            grp.create_dataset("Note", data=langchain_chunk_embed(text), compression="gzip")