import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")

def longformerize(text: str):
    # tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    # global attention: CLS token only
    global_attention_mask = torch.zeros_like(inputs["attention_mask"])
    global_attention_mask[:, 0] = 1
    
    # forward
    outputs = model(**inputs, global_attention_mask=global_attention_mask)
    last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
    
    # mean pool over valid tokens
    embedding = (last_hidden * inputs["attention_mask"].unsqueeze(-1)).sum(1)
    embedding /= inputs["attention_mask"].sum(1, keepdim=True)
    
    return embedding.squeeze().detach()

# Example
# print(longformerize("Patient reports chest pain and shortness of breath."))
