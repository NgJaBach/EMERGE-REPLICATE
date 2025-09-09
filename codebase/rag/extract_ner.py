import pandas as pd
import numpy as np
import math
from ..utils.bgem3 import *

df = pd.read_csv("D:/Lab/Research/EMERGE-REPLICATE/preprocessing-bach/processed/ehr.csv")

cat_col = df.columns[5:-12]
num_col = df.columns[-12:]

col_mean = df[num_col].mean()
col_std = df[num_col].std()

for idx, row in df.iterrows():
    record = ""
    entities = []

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
            entities.append(cat)
    
    for c in num_col:
        if math.isnan(row[c]):
            continue
        z_score = (row[c] - col_mean[c]) / col_std[c]
        if z_score > 2:
            entities.append(f"{c} too high")
        elif z_score < -2:
            entities.append(f"{c} too low")
    
    print(entities)
    
    # all_e = []
    # for e in entities:
    #     kg_match = cosine_filter(e, threshold=0.6)
    #     all_e = set(all_e + match)