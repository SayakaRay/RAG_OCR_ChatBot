import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
token_true_id = tokenizer.convert_tokens_to_ids("yes")
token_false_id = tokenizer.convert_tokens_to_ids("no")
max_length = 8192

# Format instruction
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

# Format the instruction with query and document
def format_instruction(query, doc, instruction=INSTRUCTION):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

# Process inputs for the model
def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

# Compute scores for a query and a list of documents
@torch.no_grad()
def compute_scores(query, docs):
    pairs = [format_instruction(query, doc) for doc in docs]
    inputs = process_inputs(pairs)
    logits = model(**inputs).logits[:, -1, :]
    scores = torch.nn.functional.log_softmax(
        torch.stack([
            logits[:, token_false_id],
            logits[:, token_true_id]
        ], dim=1), dim=1
    )[:, 1].exp()
    return scores.tolist()

# Test
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
