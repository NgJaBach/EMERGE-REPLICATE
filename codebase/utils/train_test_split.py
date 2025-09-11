import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

pd.set_option('display.max_columns', None)

RANDOM_STATE = 42  # for reproducibility

def data_split():
    df = pd.read_csv('D:/Lab/Research/EMERGE-REPLICATE/preprocessing-bach/processed/ehr.csv')

    pat = (
        df.groupby('PatientID', as_index=False)
        .agg(
            Outcome=('Outcome','first'),
            Readmission=('Readmission','first')
        )
    )

    pat['joint'] = pat['Outcome'].astype(int)*2 + pat['Readmission'].astype(int)
    print("Patient counts per joint class (O*2+R):", Counter(pat['joint']))

    # 1) Hold out 20% for TEST
    pat_trainval, pat_test = train_test_split(
        pat,
        test_size=0.20,
        stratify=pat['joint'],
        random_state=RANDOM_STATE
    )

    # 2) From the remaining 80%, carve out 12.5% as VAL  (12.5% of 80% = 10% overall)
    pat_train, pat_val = train_test_split(
        pat_trainval,
        test_size=0.125,
        stratify=pat_trainval['joint'],
        random_state=RANDOM_STATE
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

    def summarize_split(name, df_rows, df_pat):
        # Row-level percentages
        o_row = df_rows['Outcome'].mean()
        r_row = df_rows['Readmission'].mean()
        # Patient-level percentages
        o_pat = df_pat['Outcome'].mean()
        r_pat = df_pat['Readmission'].mean()
        print(f"{name} — Outcome: rows={o_row:.3%}, patients={o_pat:.3%} | "
            f"Readmission: rows={r_row:.3%}, patients={r_pat:.3%}")

    summarize_split("Train", train_df, pat_train)
    summarize_split("Val",   val_df,   pat_val)
    summarize_split("Test",  test_df,  pat_test)

    return train_ids, val_ids, test_ids