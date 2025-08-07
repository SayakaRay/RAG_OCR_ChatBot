# RAG_Tools/reranker.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
# --- จุดที่แก้ไข: บังคับให้โมเดลทำงานบน CPU ---
model.to('cpu')

@torch.no_grad()
def compute_scores(query, docs):
    inputs = tokenizer(
        [[query, doc] for doc in docs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    # ไม่จำเป็นต้อง .to(model.device) อีกต่อไปเพราะโมเดลอยู่บน CPU แล้ว
    outputs = model(**inputs)
    scores = outputs.logits.squeeze(-1)
    return scores.cpu().tolist() # .cpu() เพื่อความแน่นอน