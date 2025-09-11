import pandas as pd
import numpy as np
import math
from ..utils.bgem3 import cosine_filter
from collections import defaultdict
import pickle
from tqdm import tqdm
from ..utils.call_llm import extract_note, create_summary
from ..utils.clinical_longformer import langchain_chunk_embed
from ..utils.train_test_split import data_split
import h5py
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Preparing datasets
df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/preprocessing-bach/processed/ehr.csv")

with open("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/disease_features_cleaned.pkl", "rb") as f:
    kg = pickle.load(f)
mapping = dict(zip(kg["node_index"], kg["mondo_name"]))

with open("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/kg_adjacency.pkl", "rb") as f:
    adj = pickle.load(f)

notes_emb = {}
with h5py.File("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/notes_embeddings.h5", "r") as h5:
    for patient_id in h5.keys():  # each group is named by patient_id
        grp = h5[patient_id]
        pid = int(grp["PatientID"][()])
        embedding = np.array(grp["Note"])
        notes_emb[pid] = embedding
# print(notes_emb.keys())
# print(notes_emb[91199].shape) # 768

# Preprocess EHR data to extract entities
cat_col = df.columns[5:-12]
num_col = df.columns[-12:]

col_mean = df[num_col].mean()
col_std = df[num_col].std()

entities = defaultdict(list)

for idx, row in tqdm(df.iterrows(), total=len(df)):
    record = ""
    PatientID = row["PatientID"]

    if row["Sex"] == 1:
        record += "Gender: Male\n"
    else:
        record += "Gender: Female\n"
    record += f"Age: {row['Age']}\n"
    
    for c in cat_col:
        if row[c] == 1:
            cat = c
            if "Glascow coma scale total" not in cat:
                for i in range(0, 30, 1):
                    cat = cat.replace(f"->{i}.0", " : ")
                    cat = cat.replace(f"->{i}", " : ")
            cat = cat.replace("->", " : ")
            entities[PatientID].append(cat)
    
    for c in num_col:
        if math.isnan(row[c]):
            continue
        z_score = (row[c] - col_mean[c]) / col_std[c]
        if z_score > 2:
            entities[PatientID].append(f"{c} too high")
        elif z_score < -2:
            entities[PatientID].append(f"{c} too low")

# Match entities to knowledge graph
patients = list(df["PatientID"].unique())

train_ids, val_ids, test_ids = data_split()

h5_train = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete/train.h5"
h5_val = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete/val.h5"
h5_test = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete/test.h5"
str_dtype = h5py.string_dtype(encoding="utf-8")

with h5py.File(h5_train, "w") as f:
    pass
with h5py.File(h5_val, "w") as f:
    pass
with h5py.File(h5_test, "w") as f:
    pass

def _to_float32_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if isinstance(x, np.ndarray):
        return x.astype("float32", copy=False)
    raise TypeError(f"Expected tensor/ndarray, got {type(x)}")

def store_patient(h5_path, p, ehr, target, notes, summary):
    with h5py.File(h5_path, "a") as h5:
        grp = h5.create_group(str(p))
        grp.create_dataset("PatientID", data=np.asarray(p, dtype="int64"))
        grp.create_dataset("X", data=ehr, compression="gzip")
        grp.create_dataset("Note", data=_to_float32_array(notes), compression="gzip")
        grp.create_dataset("Summary", data=_to_float32_array(summary), compression="gzip")
        grp.create_dataset("Y", data=np.asarray(target, dtype="int8"))

def get_summary(p):
    entities[p] = list(set(entities[p]))
    summary_entities = ""
    summary_nodes = ""
    summary_edges = ""
    nodes = []
    for e in entities[p]:
        summary_entities += e + ", "
        idx = cosine_filter(None, e, threshold=0.6, top_k=3)
        nodes.extend(idx)
    summary_entities = summary_entities[:-2]

    nodes = list(set(nodes))
    
    for n in nodes:
        summary_nodes += kg.iloc[n]["Diseases"] + ", "
        node_x = kg.iloc[n]["node_index"]
        for connect_to in adj[n]:
            rela = connect_to[1]
            node_y = connect_to[0]
            if node_y not in kg["node_index"].values:
                continue
            e = "(" + mapping[node_x] + ", " + str(rela) + ", " + mapping[node_y] + ")"
            # print(e)
            summary_edges += e + ", "
    summary_edges = summary_edges[:-2]
    summary_nodes = summary_nodes[:-2]
    summary_notes = extract_note(notes=notes_emb[p])

    summary = create_summary(summary_entities, summary_notes, summary_nodes, summary_edges)
    return langchain_chunk_embed(summary)

# summaries = defaultdict(list)
# for p in tqdm(patients, total=len(patients), desc="Generating summaries"):
#     summaries[p] = get_summary(p)

feature_cols = [c for c in df.columns if c not in ["PatientID","Outcome","Readmission"]]
target_map = df.groupby("PatientID")[["Outcome","Readmission"]].first()

for p in tqdm(patients):
    data_ehr = df.loc[df["PatientID"] == p, feature_cols].to_numpy()
    data_notes = notes_emb[p]
    data_summary = data_notes  # default to notes if summary fails
    # data_summary = summaries[p]
    outcome, readm = target_map.loc[p].astype(int)
    data_target = (int(outcome), int(readm))

    if p in train_ids:
        h5_path = h5_train
    elif p in val_ids:
        h5_path = h5_val
    else:
        h5_path = h5_test
    store_patient(h5_path, p, data_ehr, data_target, data_notes, data_summary)
