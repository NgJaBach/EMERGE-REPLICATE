# save as filter_both_modalities.py (same folder as ehr.csv and notes.csv)
import os
import re
import pandas as pd

EHR_FILE = "raw/ehr.csv"
NOTES_FILE = "raw/notes.csv"
OUTDIR = "processed"

def _normalize_pid(s):
    # Extract leading digits only: "100_1" -> "100", "100-1" -> "100"
    return s.astype(str).str.extract(r"^(\d+)")[0]

def main():
    ehr = pd.read_csv(EHR_FILE)
    notes = pd.read_csv(NOTES_FILE)

    patientid = "PatientID"

    # Work on copies, never mutate your originals
    ehr = ehr.copy()
    notes = notes.copy()

    ehr["_PID_"] = _normalize_pid(ehr[patientid])
    notes["_PID_"] = _normalize_pid(notes[patientid])

    ehr_pids = set(ehr["_PID_"])
    notes_pids = set(notes["_PID_"])
    keep = ehr_pids & notes_pids

    ehr_out_df = ehr[ehr["_PID_"].isin(keep)].drop(columns=["_PID_"])
    notes_out_df = notes[notes["_PID_"].isin(keep)].drop(columns=["_PID_"])

    os.makedirs(OUTDIR, exist_ok=True)
    ehr_out_path = os.path.join(OUTDIR, "ehr.csv")
    notes_out_path = os.path.join(OUTDIR, "notes.csv")

    ehr_out_df.to_csv(ehr_out_path, index=False)
    notes_out_df.to_csv(notes_out_path, index=False)

    print(f"Patients with both modalities: {len(keep)}")
    print(f"EHR rows:  {len(ehr)} -> {len(ehr_out_df)}")
    print(f"Notes rows:{len(notes)} -> {len(notes_out_df)}")
    print(f"Wrote: {ehr_out_path} and {notes_out_path}")

if __name__ == "__main__":
    main()
