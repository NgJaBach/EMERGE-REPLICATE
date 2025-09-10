import pandas as pd
import numpy as np
import math
from ..utils.bgem3 import cosine_filter
from collections import defaultdict
import pickle
from tqdm import tqdm
from ..utils.call_llm import extract_note
from ..utils.clinical_longformer import longformerize
import h5py

df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/preprocessing-bach/processed/ehr.csv")

with open("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/disease_features_cleaned.pkl", "rb") as f:
    kg = pickle.load(f)

with open("D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/kg_adjacency.pkl", "rb") as f:
    adj = pickle.load(f)

notes_df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/preprocessing-bach/processed/notes.csv")

mapping = dict(zip(kg["node_index"], kg["mondo_name"]))

notes = defaultdict(list)
for idx, row in notes_df.iterrows():
    PatientID = row["PatientID"]
    note = row["Text"]
    notes[PatientID].append(note)

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

patients = list(df["PatientID"].unique())

h5_path = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete_patients_data.h5"
str_dtype = h5py.string_dtype(encoding="utf-8")

def store_patient(h5, p, ehr, target, notes, summary):
    if str(p) in h5:
        del h5[str(p)]
    grp = h5.create_group(str(p))
    grp.create_dataset("EHR", data=ehr, compression="gzip")
    grp.create_dataset("Outcome_Readmission", data=np.array(target, dtype="int8"))
    grp.create_dataset("Notes", data=notes.astype("float32"), compression="gzip")
    grp.create_dataset("Summary", data=summary.astype("float32"), compression="gzip")

def get_summary(p):

    return ""

    entities[p] = list(set(entities[p]))
    summary_entities = ""
    summary_nodes = ""
    nodes = []
    for e in entities[p]:
        summary_entities += e + ", "
        idx = cosine_filter(e, threshold=0.6, top_k=3)
        nodes.extend(idx)
    summary_entities = summary_entities[:-2]

    nodes = list(set(nodes))
    summary_nodes = ", "
    
    summary_edges = ""
    for n in nodes:
        node_x = kg.iloc[n]["node_index"]
        for connect_to in adj[n]:
            rela = connect_to[1]
            node_y = connect_to[0]
            if node_y not in kg["node_index"].values:
                continue
            e = "(" + mapping[node_x] + ", " + str(rela) + ", " + mapping[node_y] + ")"
            print(e)
            summary_edges += e + ", "
    summary_edges = summary_edges[:-2]

    summary_notes = ""

    # return create_summary(summary_entities, summary_notes, summary_nodes, summary_edges)

feature_cols = [c for c in df.columns if c not in ["PatientID","Outcome","Readmission"]]
target_map = df.groupby("PatientID")[["Outcome","Readmission"]].first()

with h5py.File(h5_path, "a") as h5:
    for p in tqdm(patients):
        data_ehr = df.loc[df["PatientID"] == p, feature_cols].to_numpy()
        data_notes = longformerize("\n".join(notes[p]))
        data_summary = get_summary(p)
        outcome, readm = target_map.loc[p].astype(int)
        data_target = (int(outcome), int(readm))

        store_patient(h5, p, data_ehr, data_target, data_notes, data_summary)

        # row = {
        #     'PatientID': p,
        #     'Target': data_target,
        #     'EHR': data_ehr, # data
        #     'Notes': data_notes, # embedding
        #     'Summary': data_summary, # embedding
        # }
        # print(f"PatientID: {p}\nEHR shape: {data_ehr.shape}\nNotes shape: {data_notes.shape}\nSummary: {data_summary}\nTarget: {data_target}")



