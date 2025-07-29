from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from pathlib import Path
import os
import json
import re
import io
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

from embeddings_2 import embed_texts, embed_queries
from faiss_index import FaissIndex
from reranker2 import compute_scores
from chunk_text import split_text_with_langchain
from bm25_index import BM25Retriever
from OCR.storage import S3Storage
from OCR.main import pipeline as ocr_pipeline

# ─────────────────────────────────────────────────────────────
# 🔧 Settings & Initialization
# ─────────────────────────────────────────────────────────────
load_dotenv()

TEXT_PATH = Path("./text_document/all_pdf.txt")
client = OpenAI()

app = FastAPI()

# Global state
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
def rebuild_index() -> bool:
    global documents, doc_embeddings, faiss_index, bm25

    if not TEXT_PATH.exists():
        return False

    print(f"⌛ Rebuilding index from {TEXT_PATH}...")
    text = TEXT_PATH.read_text(encoding="utf-8")
    documents = split_text_with_langchain(text, chunk_size=1024, chunk_overlap=100)
    doc_embeddings = embed_texts(documents)

    faiss_index = FaissIndex()
    faiss_index.build_index(doc_embeddings)

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
    try:
        print(f"📥 Uploading {file.filename} for user {username} in project {project_id}...")
        file_ext = Path(file.filename).suffix.lower().lstrip(".")
        if file_ext not in ["pdf", "jpg", "jpeg", "png", "pptx"]:
            return {"error": "Unsupported file type. Only PDF, PPTX, JPG, JPEG, PNG allowed."}

        temp_dir = {
            "pdf": "./pdf_files",
            "pptx": "./pptx_files",
            "jpg": "./temp_images",
            "jpeg": "./temp_images",
            "png": "./temp_images"
        }[file_ext]
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = Path(temp_dir) / f"temp_{file.filename}"

        # Save to disk
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # OCR Processing
        try:
            await ocr_pipeline(temp_path, file_ext, username, project_id)
        except Exception as e:
            return {"error": f"OCR failed: {str(e)}"}
        finally:
            if temp_path.exists():
                os.remove(temp_path)
                print(f"🧹 Deleted temp file: {temp_path}")

        # Rebuild index
        if rebuild_index():
            return {"message": f"{file.filename} uploaded, OCR complete, and index rebuilt."}
        return {"error": "OCR completed, but index rebuild failed."}

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

# ─────────────────────────────────────────────────────────────
# 🔍 Hybrid Search Endpoint
# ─────────────────────────────────────────────────────────────
@app.get("/search-hybrid")
async def search_hybrid(
    user_query: str,
    top_k: int = 10,
    top_rerank: int = 3,
    alpha: float = 0.7
):
    global documents, bm25, faiss_index

    if not documents or bm25 is None or faiss_index is None:
        return {"error": "Index not initialized. Please upload a file first."}

    # Step 1: Embed Query
    query_vector = embed_queries([user_query])

    # Step 2: FAISS Retrieval
    faiss_dist, faiss_indices = faiss_index.search(query_vector, top_k=top_k)
    faiss_ranked = faiss_indices[0].tolist()
    faiss_scores = 1 - (faiss_dist[0] / (np.max(faiss_dist[0]) + 1e-8))

    # Step 3: BM25 Retrieval
    bm25_scores = bm25.get_scores(user_query)
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)

    # Step 4: Hybrid Score Fusion
    hybrid_scores = {}
    for i, idx in enumerate(faiss_ranked):
        hybrid_scores[idx] = alpha * faiss_scores[i]
    for idx, score in enumerate(bm25_norm):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + (1 - alpha) * score

    # Step 5: Select Top Hybrid Results
    hybrid_ranked = [i for i, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    hybrid_texts = [documents[i] for i in hybrid_ranked]

    # Step 6: Reranking
    rerank_scores = compute_scores(user_query, hybrid_texts)
    reranked = [i for _, i in sorted(zip(rerank_scores, hybrid_ranked), reverse=True)[:top_rerank]]
    final_context = "\n".join([documents[i] for i in reranked])

    # Step 7: LLM Answer Generation
    prompt = f"Context:\n{final_context}\n\nQuestion: {user_query}\nAnswer:"
    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "system", "content": (
                "You are a helpful assistant. Use the context to answer clearly. Use only provided images: "
                "![Image: name](name). Do not guess. End with a pun and an emoji."
            )},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.output_text.strip()
    image_filenames = re.findall(r'!\[.*?\]\((.*?)\)', answer)

    return {
        "query": user_query,
        "generated_answer": answer,
        "image_filenames": image_filenames
    }

# ─────────────────────────────────────────────────────────────
# 📥 Image Retrieval from S3
# ─────────────────────────────────────────────────────────────
@app.get("/get-image-binary/{image_id}")
async def get_image_binary(image_id: str, username: str = Query(...), project_id: str = Query(...)):
    try:
        img_binary = s3_storage.get_image_binary(image_id, username, project_id)
        return Response(
            content=img_binary,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={image_id}.jpg"}
        )
    except Exception as e:
        return {"error": f"Failed to get image: {str(e)}"}

# ─────────────────────────────────────────────────────────────
# 🔍 API Status
# ─────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "faiss_ready": faiss_index is not None,
        "bm25_ready": bm25 is not None,
        "documents_loaded": bool(documents)
    }
