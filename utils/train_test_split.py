import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from utils.constants import *

# --- Helper: inclusive column slicing by labels ---
def cols_between(df, start_label, end_label=None):
    cols = df.columns
    start_idx = cols.get_loc(start_label)
    end_idx = len(cols) - 1 if end_label is None else cols.get_loc(end_label)
    if start_idx > end_idx:
        raise ValueError(f"{start_label!r} comes after {end_label!r} in columns")
    return cols[start_idx:end_idx + 1]

def data_split_and_impute():
    df = pd.read_csv(EHR_BACH, encoding='utf-8', low_memory=False)

    # Targets at patient level
    pat = (
        df.groupby('PatientID', as_index=False)
          .agg(Outcome=('Outcome','first'),
               Readmission=('Readmission','first'))
    )
    pat['joint'] = pat['Outcome'].astype(int)*2 + pat['Readmission'].astype(int)
    print("Patient counts per joint class (O*2+R):", Counter(pat['joint']))

    # 1) TEST = 20% (stratified on joint)
    pat_trainval, pat_test = train_test_split(
        pat, test_size=0.20, stratify=pat['joint'], random_state=RANDOM_STATE
    )

    # 2) VAL = 12.5% of remaining (i.e., ~10% overall)
    pat_train, pat_val = train_test_split(
        pat_trainval, test_size=0.125, stratify=pat_trainval['joint'], random_state=RANDOM_STATE
    )

    print("Train/Val/Test patients:", len(pat_train), len(pat_val), len(pat_test))

    train_ids = set(pat_train['PatientID'])
    val_ids   = set(pat_val['PatientID'])
    test_ids  = set(pat_test['PatientID'])

    train_df = df[df['PatientID'].isin(train_ids)].copy()
    val_df   = df[df['PatientID'].isin(val_ids)].copy()
    test_df  = df[df['PatientID'].isin(test_ids)].copy()

    for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"{name}: rows={len(d):,}, patients={d['PatientID'].nunique():,}")

    # ====== DEFINE YOUR FEATURE BLOCKS (as in your example) ======
    cat_cols = list(cols_between(
        df,
        "Capillary refill rate->0.0",
        "Glascow coma scale verbal response->3 Inapprop words"
    ))
    num_cols = list(cols_between(df, "Diastolic blood pressure", None))

    # ====== COMPUTE IMPUTATION VALUES ON TRAIN ONLY ======
    # Categorical: all NaNs -> 0 (no stats needed)
    # Numeric: per-column mean from TRAIN
    # Ensure numeric dtype for means; if some numeric cols are object due to bad parsing, coerce safely
    train_num = train_df[num_cols].apply(pd.to_numeric, errors='coerce')
    num_impute = train_num.mean()  # pandas Series indexed by column name

    # Keep for reuse elsewhere if needed
    impute_stats = {
        "numeric_means": num_impute,   # Series
        "categorical_fill_value": 0
    }

    # ====== APPLY IMPUTATION (USING TRAIN STATS) ======
    def apply_impute(d):
        d = d.copy()
        if cat_cols:
            d.loc[:, cat_cols] = d.loc[:, cat_cols].fillna(0)
        if num_cols:
            d.loc[:, num_cols] = d.loc[:, num_cols].apply(pd.to_numeric, errors='coerce')
            d.loc[:, num_cols] = d.loc[:, num_cols].fillna(num_impute)
        return d

    train_df_i = apply_impute(train_df)
    val_df_i   = apply_impute(val_df)
    test_df_i  = apply_impute(test_df)

    # ====== QUICK CHECKS AFTER IMPUTATION ======
    def summarize_split(name, df_rows, df_pat):
        o_row = df_rows['Outcome'].mean()
        r_row = df_rows['Readmission'].mean()
        o_pat = df_pat['Outcome'].mean()
        r_pat = df_pat['Readmission'].mean()
        print(
            f"{name} — Outcome: rows={o_row:.3%}, patients={o_pat:.3%} | "
            f"Readmission: rows={r_row:.3%}, patients={r_pat:.3%}"
        )

    summarize_split("Train", train_df_i, pat_train)
    summarize_split("Val",   val_df_i,   pat_val)
    summarize_split("Test",  test_df_i,  pat_test)

    return {
        "train": train_df_i,
        "val": val_df_i,
        "test": test_df_i,
    }

if __name__ == "__main__":
    out = data_split_and_impute()