import torch
from typing import List, Optional, Tuple
from FlagEmbedding import BGEM3FlagModel
import pickle
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device, trust_remote_code=True)

with open("D:/Lab/Research/EMERGE-REPLICATE/rag/curated_data/disease_features_cleaned.pkl", "rb") as f:
    df = pickle.load(f)
corpus_embs = torch.stack([torch.tensor(e) for e in df["embed"].values]).to(device)

def batch_encode(
    texts: List[str],
    batch_size: int = 64,
    max_length: int = 512,
) -> torch.Tensor:
    assert texts, "texts must be non-empty"
    np_vecs = model.encode(texts, batch_size=batch_size, max_length=max_length)["dense_vecs"]
    return torch.from_numpy(np_vecs).to(device)

def cosine_filter(query: str, threshold: float = 0.6) -> torch.Tensor:
    query_emb = batch_encode([query])[0]
    if query_emb.ndim == 1:
        query_emb = query_emb.unsqueeze(0)
    scores = (query_emb @ corpus_embs.T).squeeze(0)   # [N]
    keep = scores >= threshold
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    scores_kept = scores[idx]
    scores_sorted, order = torch.sort(scores_kept, descending=True)
    idx_sorted = idx[order]
    return idx_sorted.cpu().numpy()

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    # Encode a query and score
    # idx = cosine_filter("fluid in the lungs causing breathing difficulty", threshold=0.6)
    # matches = [df[i] for i in idx.tolist()]
    # print("Matches:", list(zip(matches, scores.tolist())))