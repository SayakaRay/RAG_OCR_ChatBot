# app.py (เวอร์ชันแสดงผล Chunks)

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import json
import re
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import traceback

# Import RAG and OCR tools
from RAG_Tools.embeddings import embed_texts, embed_queries
from RAG_Tools.faiss_index import FaissIndex
from RAG_Tools.reranker import compute_scores
from RAG_Tools.chunk_text import split_text_with_langchain
from RAG_Tools.bm25_index import BM25Retriever
from OCR.main import pipeline as ocr_pipeline
from OCR.storage import S3Storage

# ─────────────────────────────────────────────────────────────
# 🔧 Settings & Initialization
# ─────────────────────────────────────────────────────────────
load_dotenv()
client = OpenAI()
app = FastAPI()

# --- CORS Settings ---
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for RAG
documents = []
doc_embeddings = None
faiss_index = FaissIndex()
bm25 = None

s3_storage = S3Storage(
    os.getenv("AWS_ACCESS_KEY"),
    os.getenv("AWS_SECRET_KEY"),
    os.getenv("AWS_BUCKET_NAME"),
    "ap-southeast-2"
)

# ─────────────────────────────────────────────────────────────
# 📦 Rebuild Vector & BM25 Indexes
# ─────────────────────────────────────────────────────────────
def rebuild_index(ocr_output_path: Path) -> bool:
    global documents, doc_embeddings, faiss_index, bm25

    if not ocr_output_path or not ocr_output_path.exists():
        print(f"❌ Indexing failed: File not found at {ocr_output_path}")
        return False

    print(f"⌛ Rebuilding index from {ocr_output_path.name}...")
    text_with_images = ocr_output_path.read_text(encoding="utf-8")
    
    # Pre-process text to remove Base64 image data
    def caption_replacer(match):
        caption = match.group(1)
        return f"\n(Image Reference: {caption})\n"
    text_for_embedding = re.sub(r'!\[(.*?)\]\(data:image/png;base64,.*?\)', caption_replacer, text_with_images, flags=re.DOTALL)
    
    # Chunk the cleaned text
    documents_raw = split_text_with_langchain(text_for_embedding, chunk_size=1024, chunk_overlap=100)
    
    # Filter out any empty or whitespace-only chunks
    documents = [doc for doc in documents_raw if doc and doc.strip()]
    
    # --- จุดที่เพิ่มเข้ามา: แสดงผล Chunks ---
    print("\n" + "="*25 + " Filtered Chunks " + "="*25)
    if documents:
        for i, chunk in enumerate(documents):
            print(f"--- Chunk {i+1} ---\n{chunk}\n")
    else:
        print("No valid chunks to display.")
    print("="*68 + "\n")
    # ------------------------------------
    
    if not documents:
        print("❌ Indexing failed: No valid text chunks were found after filtering.")
        return True 
    
    doc_embeddings = embed_texts(documents)
    if doc_embeddings is None or len(doc_embeddings) == 0:
        print("❌ Indexing failed: Embedding process returned no vectors.")
        return True

    faiss_index = FaissIndex(); faiss_index.build_index(np.array(doc_embeddings))
    bm25 = BM25Retriever(documents)
    print(f"✅ Rebuilt index with {len(documents)} chunks.")
    return True

# ─────────────────────────────────────────────────────────────
# 📤 Upload & Process File Endpoint
# ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    username: str = Query(...),
    project_id: str = Query(...)
):
    ocr_output_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read()); temp_path = Path(tmp.name)
    try:
        print(f"📥 Uploading {file.filename} for user {username} in project {project_id}...")
        file_type = temp_path.suffix.lower().lstrip(".")
        ocr_output_path = await ocr_pipeline(
            str(temp_path), 
            file_type, 
            username, 
            project_id,
            original_filename=file.filename
        )
    except Exception as e:
        print(f"An error occurred during the upload or OCR process.")
        traceback.print_exc()
        return {"error": f"Upload/OCR failed: {str(e)}"}
    finally:
        os.unlink(temp_path)

    if ocr_output_path and rebuild_index(ocr_output_path):
        return {"message": f"{file.filename} uploaded, OCR complete, and index rebuilt."}
    
    return {"error": "OCR completed, but index rebuild failed. Please check server logs."}

# ─────────────────────────────────────────────────────────────
# 🔍 Hybrid Search & Other Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/search-hybrid")
async def search_hybrid(user_query: str, top_k: int = 10, top_rerank: int = 3, alpha: float = 0.7):
    global documents, bm25, faiss_index
    if not documents: 
        return {"error": "Index not initialized or document has no text. Please upload a file with text content."}
    
    query_vector = np.array(embed_queries([user_query]))
    if query_vector.size == 0:
        return {"error": "Could not generate query embedding."}

    faiss_dist, faiss_indices = faiss_index.search(query_vector, top_k=min(top_k, len(documents)))
    
    faiss_ranked = faiss_indices[0].tolist()
    faiss_scores = 1 - (faiss_dist[0] / (np.max(faiss_dist[0]) + 1e-9))

    bm25_scores = np.array(bm25.get_scores(user_query))
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)

    hybrid_scores = {}
    for i, idx in enumerate(faiss_ranked):
        hybrid_scores[idx] = alpha * faiss_scores[i]
    for idx, score in enumerate(bm25_norm):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + (1 - alpha) * score
        
    hybrid_ranked_indices = [i for i, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    hybrid_texts = [documents[i] for i in hybrid_ranked_indices]

    rerank_scores = compute_scores(user_query, hybrid_texts)
    reranked_results = sorted(zip(rerank_scores, hybrid_ranked_indices), reverse=True, key=lambda x: x[0])[:top_rerank]
    
    final_context_docs = [documents[i] for _, i in reranked_results]
    final_context = "\n\n".join(final_context_docs)
    
    prompt = f"Context:\n{final_context}\n\nQuestion: {user_query}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided context."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    
    return {"generated_answer": answer}

@app.get("/get-image-binary/{image_id}")
async def get_image_binary(image_id: str, username: str = Query(...), project_id: str = Query(...)):
    try:
        img_binary = s3_storage.get_image_binary(image_id, username, project_id)
        return Response(content=img_binary, media_type="image/jpeg", headers={"Content-Disposition": f"inline; filename={image_id}.jpg"})
    except Exception as e:
        return {"error": f"Failed to get image: {str(e)}"}

@app.get("/status")
async def status():
    return {
        "faiss_ready": faiss_index.index.ntotal > 0,
        "bm25_ready": bm25 is not None,
        "documents_loaded": bool(documents)
    }