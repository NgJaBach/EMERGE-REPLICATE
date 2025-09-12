import torch
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer").to(device).eval()

def longformerize(text: str):
    with torch.inference_mode():
        # tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        # always move inputs to the *model's* device
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

    return embedding.squeeze(0).cpu()

def langchain_chunk_embed(
    text: str,
    chunk_size: int = 4096,
    chunk_overlap: int = 256,
    aggregate: str = "weighted"  # "mean" or "weighted"
):

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                              chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    embeddings = []
    lengths = []
    for c in chunks:
        emb = longformerize(c)  # truncation inside longformerize if slightly too big
        embeddings.append(emb)
        lengths.append(len(c))
    stack = torch.stack(embeddings) if embeddings else torch.zeros(1)
    if aggregate == "mean":
        doc_emb = stack.mean(0)
    else:  # weighted
        w = torch.tensor(lengths, dtype=torch.float32).unsqueeze(1)
        doc_emb = (stack * w).sum(0) / w.sum()
    return doc_emb
    return {
        "document_embedding": doc_emb,
        "chunk_embeddings": stack,
        "chunks": chunks,
        "num_chunks": len(chunks)
    }

if __name__ == "__main__":
    txt = "Example sentence. " * 800
    out = langchain_chunk_embed(txt)
    print(out.shape)

print("cuda available:", torch.cuda.is_available())
print("model device:", next(model.parameters()).device)