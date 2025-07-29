import json
import numpy as np
import matplotlib.pyplot as plt
from embeddings_2 import embed_queries, embed_texts
from faiss_index import FaissIndex
from reranker3 import compute_scores
from chunk_text import split_text_with_langchain
from bm25_index import BM25Retriever
from ocr import get_all_pdf
import os
from collections import defaultdict
import matplotlib.font_manager as fm

# Load and Prepare Data
PDF_PATH = ""
TEXT_PATH = "./text_document/biology.txt"
EVAL_SET = "./eval_data4.json"

if not os.path.exists(TEXT_PATH):
    get_all_pdf(PDF_PATH, output_path=TEXT_PATH)

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    big_text = f.read()

documents = split_text_with_langchain(big_text, chunk_size=1024, chunk_overlap=100)
print(f"✅ Loaded {len(documents)} chunks")

doc_embeddings = embed_texts(documents)
faiss_index = FaissIndex()
faiss_index.build_index(doc_embeddings)
bm25 = BM25Retriever(documents)

with open(EVAL_SET, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

# Metrics
def precision_at_k(pred, gold, k):
    return len(set(pred[:k]) & set(gold)) / k

def recall_at_k(pred, gold, k):
    return len(set(pred[:k]) & set(gold)) / len(gold) if gold else 0

def mrr(pred, gold):
    for rank, pid in enumerate(pred, start=1):
        if pid in gold:
            return 1.0 / rank
    return 0.0

def f1_score(p, r):
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

# --- Evaluation Loop ---
results = defaultdict(list)
all_metrics = []
top_k_per_query = []

for idx, item in enumerate(eval_data):
    query = item["query"]
    relevant = item["relevant"]
    query_vec = embed_queries([query])

    # FAISS
    faiss_distances, faiss_indices = faiss_index.search(query_vec, top_k=10)
    faiss_ranked = faiss_indices[0].tolist()
    faiss_scores = 1 - (faiss_distances[0] / (np.max(faiss_distances[0]) + 1e-8))

    # BM25
    bm25_scores = bm25.get_scores(query)
    bm25_ranked = np.argsort(bm25_scores)[::-1][:10].tolist()
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)

    # Hybrid
    hybrid_scores = {}
    for i, idx_ in enumerate(faiss_ranked):
        hybrid_scores[idx_] = 0.7 * faiss_scores[i]
    for idx_, bm25_score in enumerate(bm25_norm):
        hybrid_scores[idx_] = hybrid_scores.get(idx_, 0) + 0.3 * bm25_score

    hybrid_ranked = [idx for idx, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]]

    # Rerank Hybrid
    rerank_texts = [documents[i] for i in hybrid_ranked]
    rerank_scores = compute_scores(query, rerank_texts)
    reranked_hybrid = [i for _, i in sorted(zip(rerank_scores, hybrid_ranked), reverse=True)[:3]]

    # Store rankings for this query
    top_k_per_query.append({
        "faiss": faiss_ranked,
        "bm25": bm25_ranked,
        "hybrid": hybrid_ranked,
        "rerank": reranked_hybrid
    })

    # Metrics
    metric_values = {}
    for name, ranked in zip(["FAISS", "BM25", "Hybrid", "Rerank"],
                             [faiss_ranked, bm25_ranked, hybrid_ranked, reranked_hybrid]):
        p = precision_at_k(ranked, relevant, 5)
        r = recall_at_k(ranked, relevant, 5)
        m = mrr(ranked, relevant)
        f1 = f1_score(p, r)
        results[f"{name.lower()}_p@5"].append(p)
        results[f"{name.lower()}_r@5"].append(r)
        results[f"{name.lower()}_mrr"].append(m)
        results[f"{name.lower()}_f1@5"].append(f1)
        metric_values[name] = (p, r, m, f1)

    all_metrics.append((query, metric_values))

# --- Move log writing here, after the loop ---
os.makedirs("./eval_bio_6/logs", exist_ok=True)
with open("./eval_bio_6/logs/top_k_results.txt", "w", encoding="utf-8") as log_file:
    for i, (query, metric_dict) in enumerate(all_metrics):
        log_file.write(f"=== Query {i+1} ===\n")
        log_file.write(f"Query: {query}\n")
        log_file.write(f"Relevant IDs: {eval_data[i]['relevant']}\n\n")

        top_k = top_k_per_query[i]
        for step_name, ranked in zip(["FAISS", "BM25", "Hybrid", "Rerank"],
                                     [top_k["faiss"], top_k["bm25"], top_k["hybrid"], top_k["rerank"]]):
            log_file.write(f"--- {step_name} Top-{len(ranked)} Results ---\n")
            for rank, idx in enumerate(ranked, start=1):
                snippet = documents[idx][:100].replace("\n", " ").strip()
                log_file.write(f"{rank:02d}. [ID {idx}] {snippet}...\n")
            log_file.write("\n")

        log_file.write("="*50 + "\n\n")

print("📝 Top‑K logs saved at ./eval_bio_6/logs/top_k_results.txt")

# Metrics
print("\n===== Evaluation Results =====")
for metric, scores in results.items():
    print(f"{metric}: {np.mean(scores):.3f}")

# Plot for Each Query
print("\n📊 ploting graph")
os.makedirs("./eval_bio_6/plots/qwen3", exist_ok=True)
try:
    th_font = fm.FontProperties(fname="C:/Windows/Fonts/leelawad.ttf")
except Exception:
    th_font = fm.FontProperties()

for i, (query, metric_dict) in enumerate(all_metrics):
    models = list(metric_dict.keys())
    p_scores = [metric_dict[m][0] for m in models]
    r_scores = [metric_dict[m][1] for m in models]
    mrr_scores = [metric_dict[m][2] for m in models]
    f1_scores = [metric_dict[m][3] for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))

    ax.plot(x, p_scores, marker='o', label='Precision@5')
    ax.plot(x, r_scores, marker='s', label='Recall@5')
    ax.plot(x, mrr_scores, marker='^', label='MRR')
    ax.plot(x, f1_scores, marker='d', linestyle='--', label='F1-Score@5')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontproperties=th_font)
    ax.set_title(f"Query {i+1}: {query[:50]}...", fontproperties=th_font)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"./eval_bio_6/plots/qwen3/query_{i+1}.png")
    plt.close(fig)

print("\u2705 All plots saved in ./eval_bio_6/plot")
