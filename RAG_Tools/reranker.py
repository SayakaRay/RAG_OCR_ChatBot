import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "BAAI/bge-reranker-v2-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

@torch.no_grad()
def compute_scores(query, docs):
    # Prepare input pairs
    inputs = tokenizer(
        [[query, doc] for doc in docs],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)
    scores = outputs.logits.squeeze(-1)

    return scores.tolist()

if __name__ == "__main__":
    query = "What is the capital of France?"
    docs = [
        "The capital of France is Paris.",
        "France is a country in Europe.",
        "Paris is known for its art, fashion, and culture."
    ]

    scores = compute_scores(query, docs)
    for doc, score in zip(docs, scores):
        print(f"Document: {doc}\nScore: {score:.4f}\n")
