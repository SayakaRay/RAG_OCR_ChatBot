# rerank BAAI/bge-reranker-v2-gemma
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "BAAI/bge-reranker-v2-gemma"
INSTRUCTION = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
token_true_id = tokenizer.convert_tokens_to_ids("Yes")
max_length = 1024

# Format the instruction with query and document
def format_instruction(query, doc, instruction=INSTRUCTION):
    return f"A: {query}\nB: {doc}\n{instruction}"

# Process inputs for the model
def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=True, truncation=True,
        return_attention_mask=False, return_tensors="pt", max_length=max_length
    )
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

# Compute scores for a query and a list of documents
@torch.no_grad()
def compute_scores(query, docs):
    pairs = [format_instruction(query, doc) for doc in docs]
    
    inputs = process_inputs(pairs)
    logits = model(**inputs).logits[:, -1, :]
    
    scores = logits[:, token_true_id]
    return scores.tolist()

# Test
if __name__ == "__main__":
    query = "อะไรคือเมืองหลวงของประเทศไทย?"
    docs = [
        "ประเทศไทยตั้งอยู่ในเอเชียตะวันออกเฉียงใต้",
        "กรุงเทพฯ เป็นศูนย์กลางการเมืองและเศรษฐกิจของไทย",
        "กรุงเทพมหานครคือเมืองหลวงของประเทศไทย",
        "เชียงใหม่เป็นเมืองสำคัญในภาคเหนือของไทย",
        "หมูป่าพาหมาป่ามากินหมี่ที่เยาวราช"
    ]
    
    scores = compute_scores(query, docs)

    for doc, score in zip(docs, scores):
        print(f"Document: {doc}\nScore: {score:.4f}\n")