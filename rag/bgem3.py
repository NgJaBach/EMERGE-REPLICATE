import torch
from typing import List, Optional, Tuple
from FlagEmbedding import BGEM3FlagModel

def batch_encode(
    texts: List[str],
    batch_size: int = 64,
    max_length: int = 512,
) -> torch.Tensor:
    assert texts, "texts must be non-empty"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device, trust_remote_code=True)
    np_vecs = model.encode(texts, batch_size=batch_size, max_length=max_length)["dense_vecs"]
    return torch.from_numpy(np_vecs).to(device)

def cosine_filter(
    query_emb: torch.Tensor,         # [D] or [1, D]
    corpus_embs: torch.Tensor,       # [N, D]
    threshold: float = 0.6
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query_emb.ndim == 1:
        query_emb = query_emb.unsqueeze(0)
    scores = (query_emb @ corpus_embs.T).squeeze(0)   # [N]
    keep = scores >= threshold
    idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    scores_kept = scores[idx]
    scores_sorted, order = torch.sort(scores_kept, descending=True)
    idx_sorted = idx[order]
    return idx_sorted, scores_sorted

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
        
    texts = ["pulmonary edema", "heart failure", "renal failure", "emphysema"]
    corpus = batch_encode(texts)

    # Encode a query and score
    q = batch_encode(["fluid in the lungs causing breathing difficulty"])
    idx, scores = cosine_filter(q[0], corpus, threshold=0.6)
    matches = [texts[i] for i in idx.tolist()]
    print("Matches:", list(zip(matches, scores.tolist())))