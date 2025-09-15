import os

# PRIMEKG - dataverse_files
DATAVERSE_DIR   = "D:/Lab/Research/EMERGE-REPLICATE/datasets/dataverse_files/"
PRIMEKG_KG      = DATAVERSE_DIR + "kg.csv"
PRIMEKG_DISEASE = DATAVERSE_DIR + "disease_features.csv"

GOOD_DATA = "D:/Lab/Research/EMERGE-REPLICATE/good_data/"
RECORDS      = 48 # Number of records per patient to keep
RANDOM_STATE = 42 # for reproducibility

# PART 1
NOTES_ZHU  = "D:/Lab/Research/EMERGE-REPLICATE/preprocessing-zhu/mimic-iii/data/processed/notes.csv"
EHR_ZHU    = "D:/Lab/Research/EMERGE-REPLICATE/preprocessing-zhu/mimic-iii/data/processed/ehr.csv"
NOTES_BACH = GOOD_DATA + "processed/notes.csv"
EHR_BACH   = GOOD_DATA + f"processed/ehr.csv"

os.makedirs(os.path.dirname(EHR_BACH), exist_ok=True)

TRAIN_DRAFT = GOOD_DATA + f"incomplete/train.csv"
VAL_DRAFT   = GOOD_DATA + f"incomplete/val.csv"
TEST_DRAFT  = GOOD_DATA + f"incomplete/test.csv"

os.makedirs(os.path.dirname(TRAIN_DRAFT), exist_ok=True)
os.makedirs(os.path.dirname(VAL_DRAFT),   exist_ok=True)
os.makedirs(os.path.dirname(TEST_DRAFT),  exist_ok=True)

# PART 2
KG_ADJACENCY     = GOOD_DATA + "curated/kg_adjacency.pkl"
DISEASE_FEATURES = GOOD_DATA + "curated/disease_features_cleaned.pkl"
NOTES_EMBEDDINGS = GOOD_DATA + "curated/notes_embeddings.h5"

os.makedirs(os.path.dirname(KG_ADJACENCY),     exist_ok=True)
os.makedirs(os.path.dirname(DISEASE_FEATURES), exist_ok=True)
os.makedirs(os.path.dirname(NOTES_EMBEDDINGS), exist_ok=True)

DATA_PATH = GOOD_DATA + f"complete/"

TRAIN = DATA_PATH + "train.h5"
VAL   = DATA_PATH + "val.h5"
TEST  = DATA_PATH + "test.h5"

os.makedirs(os.path.dirname(TRAIN), exist_ok=True)
os.makedirs(os.path.dirname(VAL),   exist_ok=True)
os.makedirs(os.path.dirname(TEST),  exist_ok=True)

