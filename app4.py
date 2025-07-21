from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response
from pathlib import Path
import os
import json
import re
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from embeddings_2 import embed_texts, embed_queries
from faiss_index import FaissIndex
from reranker2 import compute_scores
from chunk_text import split_text_with_langchain
from bm25_index import BM25Retriever
from OCR.storage import S3Storage
from OCR.main import pipeline as ocr_pipeline

# --- Load Env and Init ---
load_dotenv()
app = FastAPI()
client = OpenAI()

# --- Globals for search ---
documents = []
doc_embeddings = None
faiss_index = FaissIndex()
bm25 = None
TEXT_PATH = Path("./text_document/all_pdf.txt")

# --- S3 Setup ---
s3_storage = S3Storage(
    os.getenv('AWS_ACCESS_KEY'),
    os.getenv('AWS_SECRET_KEY'),
    os.getenv('AWS_BUCKET_NAME'),
    "ap-southeast-2"
)

# --- Rebuild Index Function ---
def rebuild_index():
    global documents, doc_embeddings, faiss_index, bm25
    if not TEXT_PATH.exists():
        return False
    print(f"⌛ Start Rebuilding index from {TEXT_PATH}...")
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        big_text = f.read()

    documents = split_text_with_langchain(big_text, chunk_size=1024, chunk_overlap=100)
    doc_embeddings = embed_texts(documents)

    faiss_index = FaissIndex()
    faiss_index.build_index(doc_embeddings)

    bm25 = BM25Retriever(documents)
    print(f"✅ Rebuilt index from {len(documents)} chunks.")
    return True

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split('.')[-1].lower()
        print(f"Uploading file with extension: {file_extension}")

        if file_extension not in ['pdf', 'jpg', 'jpeg', 'png', 'pptx']:
            return {"error": "Unsupported file type. Only PDF, JPG, JPEG, PNG, and PPTX are allowed."}

        # Create temp path
        temp_folder = {
            "pdf": "./pdf_files",
            "pptx": "./pptx_files",
            "jpg": "./temp_images",
            "jpeg": "./temp_images",
            "png": "./temp_images"
        }[file_extension]

        os.makedirs(temp_folder, exist_ok=True)
        temp_path = Path(temp_folder) / f"temp_{file.filename}"

        # Save uploaded file to temp
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run OCR pipeline
        try:
            await ocr_pipeline(temp_path, file_extension)
        except Exception as e:
            return {"error": f"OCR failed: {str(e)}"}
        finally:
            if temp_path.exists():
                os.remove(temp_path)
                print(f"Deleted temp file: {temp_path}")

        # Rebuild FAISS + BM25 index
        if rebuild_index():
            return {"message": f"{file.filename} uploaded, OCR complete, and index rebuilt."}
        else:
            return {"error": "OCR done, but failed to rebuild index."}

    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}    

@app.get("/search-hybrid")
async def search_hybrid(user_query: str, top_k: int = 10, top_rerank: int = 3, alpha: float = 0.7):
    global documents, bm25, faiss_index

    if documents is None or bm25 is None or faiss_index is None:
        return {"error": "Index not initialized. Please upload a file first."}

    # Step 1: Embed Query
    query_vector = embed_queries([user_query])

    # Step 2: FAISS Retrieval
    faiss_distances, faiss_indices = faiss_index.search(query_vector, top_k=top_k)
    faiss_ranked = faiss_indices[0].tolist()
    faiss_scores = 1 - (faiss_distances[0] / (np.max(faiss_distances[0]) + 1e-8))

    # Step 3: BM25 Retrieval
    bm25_scores = bm25.get_scores(user_query)
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)

    # Step 4: Combine Hybrid Scores
    hybrid_scores = {}
    for i, idx in enumerate(faiss_ranked):
        hybrid_scores[idx] = alpha * faiss_scores[i]
    for idx, score in enumerate(bm25_norm):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + (1 - alpha) * score

    # Step 5: Select Top-K Hybrid Docs
    hybrid_ranked = [idx for idx, _ in sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    hybrid_texts = [documents[i] for i in hybrid_ranked]

    # Step 6: Rerank Top-K Hybrid Docs
    rerank_scores = compute_scores(user_query, hybrid_texts)
    reranked_hybrid = [i for _, i in sorted(zip(rerank_scores, hybrid_ranked), reverse=True)[:top_rerank]]
    reranked_texts = [documents[i] for i in reranked_hybrid]
    context = "\n".join(reranked_texts)

    # Step 7: Generate Answer
    prompt = f"Context:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    response = client.responses.create(
        model="gpt-4o-mini-2024-07-18",
        input=[
            {"role": "system", "content": (
                "You are a helpful assistant. Answer this question using the following context, by providing necessary images. "
                "Image codes should be in the format: ![Image: name](name). No file types. DO NOT make up images. "
                "If the image is not found, do not mention any images . End with a pun and an emoji."
            )},
            {"role": "user", "content": prompt}
        ],
    )

    answer = response.output_text.strip()
    print(f"Generated Answer: {answer}")
    image_filenames = re.findall(r'!\[.*?\]\((.*?)\)', answer)

    return {
        "query": user_query,
        "top_k": top_k,
        "top_rerank": top_rerank,
        "alpha": alpha,
        "generated_answer": answer,
        "image_filenames": image_filenames,
        "reranked_top_docs": [
            {
                "document": documents[i],
                "score": float(score)
            }
            for i, score in sorted(zip(reranked_hybrid, rerank_scores), key=lambda x: reranked_hybrid.index(x[0]))
        ]
    }

@app.get("/status")
async def status():
    return {
        "faiss_ready": faiss_index is not None,
        "bm25_ready": bm25 is not None,
        "documents_loaded": documents is not None and len(documents) > 0
    }

@app.get("/get-image-binary/{image_id}")
async def get_image_binary(image_id: str):
    try:
        # ดึงรูปภาพจาก S3 เป็น binary
        img_binary = s3_storage.get_image_binary(image_id)
        
        # ส่งกลับเป็น binary response
        return Response(
            content=img_binary,
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename={image_id}.jpg"}
        )
    except Exception as e:
        return {"error": f"Failed to get image: {str(e)}"}
