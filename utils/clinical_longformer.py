import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer").to(device).eval()

def longformerize(text: str):
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True,
        )
        dev = next(model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        # global attention: CLS token only
        global_attention_mask = torch.zeros_like(inputs["attention_mask"], device=device)
        global_attention_mask[:, 0] = 1
        
        # forward
        outputs = model(**inputs, global_attention_mask=global_attention_mask)
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
        
        # mean pool over valid tokens
        embedding = (last_hidden * inputs["attention_mask"].unsqueeze(-1)).sum(1)
        embedding /= inputs["attention_mask"].sum(1, keepdim=True)

    return embedding.squeeze(0).to(dtype=torch.float32, device="cpu").numpy()

def langchain_chunk_embed(
    text: str,
    chunk_size: int = 4096, # limit is 4096 for longformer
    chunk_overlap: int = 256,
    aggregate: str = "weighted"  # "mean" or "weighted"
    ):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    embs = [longformerize(c) for c in chunks]             # list of (hidden,) np.float32
    stack = np.stack(embs, axis=0).astype(np.float32)     # [n_chunks, hidden]

    if aggregate == "mean":
        doc = stack.mean(axis=0)
    else:
        lengths = np.array([len(c) for c in chunks], dtype=np.float32)  # char-length weights
        wsum = lengths.sum()
        if wsum <= 0:
            doc = stack.mean(axis=0)
        else:
            doc = (stack * lengths[:, None]).sum(axis=0) / wsum

    return doc.astype(np.float32)

def first_n_words(text: str, n: int = 256) -> str:
    # Splits on any whitespace, stops after n splits; doesn’t scan the whole string.
    parts = text.split(None, n)  # None => any whitespace; n => at most n splits
    return " ".join(parts[:n])

def plain_truncate(text: str, max_length: int = 256):
    return longformerize(first_n_words(text, max_length))

if __name__ == "__main__":
    txt = "Example sentence. " * 800
    out = langchain_chunk_embed(txt)
    print(out.shape)

print("cuda available:", torch.cuda.is_available())
print("model device:", next(model.parameters()).device)