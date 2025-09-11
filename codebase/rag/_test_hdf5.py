import h5py
import numpy as np
import torch

def test_hdf5(h5_path):
    with h5py.File(h5_path, "r") as h5:
        patient_ids = list(h5.keys())          # all patient groups
        print("N patients:", len(patient_ids))
        first = patient_ids[0]
        grp = h5[first]

        ehr = grp["EHR"][:]                    # 2D (R, D)
        target = grp["Outcome_Readmission"][:] # shape (2,)
        notes_emb = grp["Notes"][:]   # 1D
        summary_emb = grp["Summary"][:] if "Summary" in grp else None

        print(first, ehr.shape, target, notes_emb.shape, 
            None if summary_emb is None else summary_emb.shape)

def test_hdf5_2(h5_path):
    data = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            entry = {}
            grp = f[key]
            print(grp.keys())
            for k in grp.keys():
                entry[k] = grp[k][()]
            data.append(entry)
            print(entry['PatientID'])
    
    # index = 0
    # print(data[index])
    # pid = data[index]['PatientID']
    # x_ehr = torch.tensor(data[index]['X']) # preprocessed data
    # x_note = torch.tensor(data[index]['Note']) # embedding
    # # x_summary = torch.tensor(self.data[index]['Summary']) # embedding
    # y = torch.tensor(data[index]['Y'])

    # print(x_ehr.shape, x_note.shape, y.shape)
    # y_outcome = y[0]
    # y_readmission = y[1]
    # print(data)

h5_train = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete/train.h5"
h5_val = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete/val.h5"
h5_test = "D:/Lab/Research/EMERGE-REPLICATE/codebase/rag/curated_data/complete/test.h5"

test_hdf5_2(h5_train)
test_hdf5_2(h5_val)
test_hdf5_2(h5_test)