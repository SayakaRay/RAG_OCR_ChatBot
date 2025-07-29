from fastapi import FastAPI, Query
from embeddings import embed_texts, embed_queries
from faiss_index import FaissIndex
from openai import OpenAI
from dotenv import load_dotenv
from reranker import compute_scores
from chunk_text import split_text_with_langchain
from ocr import get_all_pdf 
import os

# --- Load Environment Variables ---
load_dotenv()

client = OpenAI()
app = FastAPI()

# --- OCR and Chunking Pipeline ---

PDF_PATH = "./pdf_files/anime.pdf"
TEXT_PATH = "./text_document/ocr_all_pdf.txt"

# 1. Run OCR if output doesn't already exist
if not os.path.exists(TEXT_PATH):
    print(f"Running OCR on {PDF_PATH}...")
    get_all_pdf(PDF_PATH, output_path=TEXT_PATH)

# 2. Read OCR result
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    big_text = f.read()

# 3. Chunk text
documents = split_text_with_langchain(big_text, chunk_size=1024, chunk_overlap=100)
print(f"✅ Chunked {len(documents)} text segments.")

# 4. Embed documents
doc_embeddings = embed_texts(documents)

# 5. Build FAISS index
faiss_index = FaissIndex()
faiss_index.build_index(doc_embeddings)
print("✅ FAISS index built.")

# --- FastAPI Endpoints ---

@app.get("/retrieval")
async def retrieval(user_query: str, top_k: int = 3):
    query_embedding = embed_queries([user_query])
    distances, indices = faiss_index.search(query_embedding, top_k=top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        results.append({
            "rank": rank,
            "document": documents[idx],
            "distance": float(distances[0][rank - 1])
        })

    return {
        "query": user_query,
        "top_k": top_k,
        "matches": results
    }


@app.get("/generate")
async def generate(user_query: str, top_k: int = 3 , top_rerank: int = 1):
    # Embed query
    query_embedding = embed_queries([user_query])

    # Retrieve top-k similar documents by Search FAISS
    distances, indices = faiss_index.search(query_embedding, top_k=top_k)
    retrieved_data = [documents[num] for num in indices[0]]

    # Rerank retrieved
    scores = compute_scores(user_query, retrieved_data)
    scored_docs = sorted(zip(retrieved_data, scores), key=lambda x: x[1], reverse=True) # Sort by scores
    reranked_top_docs = [doc for doc, score in scored_docs[:top_rerank]] # Take top reranked documents
    context = "\n".join(reranked_top_docs)

    # Prepare prompt for LLM
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    print("Prompt to LLM:", prompt)

    # Call LLM to generate answer
    try:
        response = client.responses.create(
            model="gpt-4o-mini-2024-07-18",
            input=[
                {"role": "system", "content": (
                    "You are a helpful assistant. Only use the context below to answer the question, "
                    "DON'T retrieve data from other sources. Also, make a pun if possible and end with an emoji."
                )},
                {"role": "user", "content": prompt}
            ],
        )
        answer = response.output_text.strip()
    except Exception as e:
        return {"error": str(e)}

    # Return reranked matches + generated answer
    return {
        "query": user_query,
        "top_k": top_k,
        "matches": [
            {"document": doc, "score": score}
            for doc, score in scored_docs[:top_k]
        ],
        "generated_answer": answer
    }