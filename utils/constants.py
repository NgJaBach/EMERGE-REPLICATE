import os

def ensure_path(THE_PATH: str):
    os.makedirs(os.path.dirname(THE_PATH), exist_ok=True)
    return THE_PATH

# PRIMEKG - dataverse_files
DATAVERSE_DIR   = "D:/Lab/Research/EMERGE-REPLICATE/datasets/dataverse_files/"
PRIMEKG_KG      = DATAVERSE_DIR + "kg.csv"
PRIMEKG_DISEASE = DATAVERSE_DIR + "disease_features.csv"

# Setup
LLM_LOG = ensure_path("D:/Lab/Research/EMERGE-REPLICATE/utils/log.txt")
GOOD_DATA = "D:/Lab/Research/EMERGE-REPLICATE/good_data/"
DATA_PATH = GOOD_DATA + f"complete/"
RECORDS      = 4
RANDOM_STATE = 42 # for reproducibility
USE_CHUNKING = False

# PART 1
NOTES_ZHU  = "D:/Lab/Research/EMERGE-REPLICATE/preprocessing-zhu/mimic-iii/data/processed/notes.csv"
EHR_ZHU    = "D:/Lab/Research/EMERGE-REPLICATE/preprocessing-zhu/mimic-iii/data/processed/ehr.csv"
NOTES_BACH = ensure_path(GOOD_DATA + "processed/notes.csv")
EHR_BACH   = ensure_path(GOOD_DATA + "processed/ehr.csv")

TRAIN_DRAFT = ensure_path(GOOD_DATA + "incomplete/train.csv")
VAL_DRAFT   = ensure_path(GOOD_DATA + "incomplete/val.csv")
TEST_DRAFT  = ensure_path(GOOD_DATA + "incomplete/test.csv")

# PART 2
KG_ADJACENCY     = ensure_path(GOOD_DATA + "curated/kg_adjacency.pkl")
DISEASE_FEATURES = ensure_path(GOOD_DATA + "curated/disease_features_cleaned.pkl")
NOTES_EMBEDDINGS = ensure_path(GOOD_DATA + "curated/notes_embeddings.h5")

SUMEMB_TRAIN = ensure_path(GOOD_DATA + "curated/summary_embeddings/train.h5")
SUMEMB_VAL   = ensure_path(GOOD_DATA + "curated/summary_embeddings/val.h5")
SUMEMB_TEST  = ensure_path(GOOD_DATA + "curated/summary_embeddings/test.h5")

TRAIN = ensure_path(DATA_PATH + "train.h5")
VAL   = ensure_path(DATA_PATH + "val.h5")
TEST  = ensure_path(DATA_PATH + "test.h5")