from fastapi import FastAPI, Query
from fastapi.responses import Response
from embeddings_2 import embed_texts, embed_queries
from faiss_index import FaissIndex
from openai import OpenAI
from dotenv import load_dotenv
from reranker2 import compute_scores
from chunk_text import split_text_with_langchain
from ocr import get_all_pdf
from bm25_index import BM25Retriever
import numpy as np
import os
import re
from storage import S3Storage
from PIL import Image
import io

# --- Load Environment Variables ---
load_dotenv()

client = OpenAI()
app = FastAPI()

# --- OCR and Chunking Pipeline ---
PDF_PATH = "./pdf_files/anime.pdf"
TEXT_PATH = "./text_document/chemistry2.txt"

# Run OCR
if not os.path.exists(TEXT_PATH):
    print(f"Running OCR on {PDF_PATH}...")
    get_all_pdf(PDF_PATH, output_path=TEXT_PATH)

# Read OCR result
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    big_text = f.read()

# Chunk text
documents = split_text_with_langchain(big_text, chunk_size=1024, chunk_overlap=100)
print(f"✅ Chunked {len(documents)} text segments.")

# Embed documents
doc_embeddings = embed_texts(documents)

# Build FAISS index
faiss_index = FaissIndex()
faiss_index.build_index(doc_embeddings)
print("✅ FAISS index built.")

# Initialize BM25
bm25 = BM25Retriever(documents)
print("✅ BM25 index initialized.")


# --- S3 Storage Setup ---
S3_ACCESS = os.getenv('AWS_ACCESS_KEY')
S3_SECRET = os.getenv('AWS_SECRET_KEY')
S3_BUCKET = os.getenv('AWS_BUCKET_NAME')
S3_REGION = "ap-southeast-2"
s3_storage = S3Storage(S3_ACCESS, S3_SECRET, S3_BUCKET, S3_REGION)

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

@app.get("/generate-hybrid")
async def generate_hybrid(user_query: str, top_k: int = 10, top_rerank: int = 3, alpha: float = 0.7):
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
                "You are a helpful assistant. Answer this question using the following context, by providing necessary images "
                "Image codes should be in the context only, if cannot find the image, say 'Image not found'."
                "image code should be in the format: ![datetime-uuid](datetime-uuid). "
                "DON'T retrieve data from other sources. Also, make a pun if possible and end with an emoji."
            )},
            {"role": "user", "content": prompt}
        ],
    )

    anwser = response.output_text.strip()
    image_filenames = re.findall(r'!\[.*?\]\((.*?)\)', anwser)
    print(f"Image filenames found: {image_filenames}")

    # Retrieve and open images from S3 using PIL (use only filename as key)
    pil_images = []
    for image_id in image_filenames:
        # Extract only the filename if path is given
        filename = os.path.basename(image_id)
        print(f"Processing image filename: {filename}")
        filename = filename.rsplit('.', 1)[0]
        print(f"Final filename to retrieve: {filename}")
        try:
            img_binary = s3_storage.get_image_binary(filename)
            img = Image.open(io.BytesIO(img_binary))
            pil_images.append({"image_id": filename, "image": img})
            print(f"Loaded image from S3: {filename}")
            
            # Uncomment the line below to display the image
            # img.show()  # Show image on your computer
        except Exception as e:
            print(f"Failed to load image {filename} from S3: {e}")

    return {
        "query": user_query,
        "top_k": top_k,
        "top_rerank": top_rerank,
        "alpha": alpha,
        "generated_answer": anwser,
        "image_filenames": image_filenames,
        "reranked_top_docs": [
            {
                "document": documents[i],
                "score": float(score)
            }
            for i, score in sorted(zip(reranked_hybrid, rerank_scores), key=lambda x: reranked_hybrid.index(x[0]))
        ]
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