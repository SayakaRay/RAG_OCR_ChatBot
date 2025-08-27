# app.py

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
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
    os.getenv("AWS_BUCKET_NAME")
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
    def caption_replacer(match):
        caption = match.group(1)
        return f"\n(Image Reference: {caption})\n"
    text_for_embedding = re.sub(r'!\[(.*?)\]\(data:image/png;base64,.*?\)', caption_replacer, text_with_images, flags=re.DOTALL)
    documents_raw = split_text_with_langchain(text_for_embedding, chunk_size=1024, chunk_overlap=100)
    documents = [doc for doc in documents_raw if doc and doc.strip()]
    print("\n" + "="*25 + " Filtered Chunks " + "="*25)
    if documents:
        for i, chunk in enumerate(documents):
            print(f"--- Chunk {i+1} ---\n{chunk}\n")
    else:
        print("No valid chunks to display.")
    print("="*68 + "\n")
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
async def upload(file: UploadFile = File(...), username: str = Query(...), project_id: str = Query(...)):
    ocr_output_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read()); temp_path = Path(tmp.name)
    try:
        print(f"📥 Uploading {file.filename} for user {username} in project {project_id}...")
        file_type = temp_path.suffix.lower().lstrip(".")
        ocr_output_path = await ocr_pipeline(
            str(temp_path), file_type, username, project_id, original_filename=file.filename
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
# 🧠 Intent Classifier (ส่วนที่เพิ่มเข้ามาใหม่)
# ─────────────────────────────────────────────────────────────
async def classify_intent(user_query: str):
    """
    ใช้ AI เพื่อวิเคราะห์ว่าผู้ใช้ต้องการ 'สรุป' หรือ 'ถามตอบ'
    """
    print(f"🧠 Classifying intent for query: '{user_query}'")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Your job is to classify the user's intent. Do they want a 'summarization' of the document, or are they asking a 'specific_question' about a detail within it?"
            }, {
                "role": "user", "content": user_query
            }],
            tools=[
                {"type": "function", "function": {"name": "summarization", "description": "The user wants a summary, overview, or general idea of the document's content (e.g., 'what is this about?', 'summarize page 5')."}},
                {"type": "function", "function": {"name": "specific_question", "description": "The user is asking for a specific detail, fact, or piece of information from the document (e.g., 'who is the author?', 'what is the chemical formula for soda ash?')."}}
            ],
            tool_choice="auto"
        )
        tool_call = response.choices[0].message.tool_calls[0]
        intent = tool_call.function.name
        print(f"✅ Intent classified as: {intent}")
        return intent
    except Exception as e:
        print(f"⚠️ Intent classification failed, defaulting to 'specific_question'. Error: {e}")
        return "specific_question"

# ─────────────────────────────────────────────────────────────
# 🔍 Upgraded Hybrid Search (Agent/Router) (ส่วนที่แก้ไข)
# ─────────────────────────────────────────────────────────────
@app.get("/search-hybrid")
async def search_hybrid(user_query: str, top_k: int = 10, alpha: float = 0.7):
    global documents, bm25, faiss_index
    if not documents: 
        return {"error": "Index not initialized. Please upload a file."}

    intent = await classify_intent(user_query)

    final_context_docs = []
    
    if intent == "summarization":
        print("🚀 Routing to: Summarization Tool")
        page_match = re.search(r'หน้า(?:ที่)?\s*(\d+)', user_query)
        if page_match:
            page_num = int(page_match.group(1))
            print(f"📚 Summarization requested for page {page_num}")
            page_pattern = re.compile(rf"## Page {page_num}\n(.*?)(?=\n## Page|\Z)", re.DOTALL)
            all_text = "\n".join(documents)
            page_content = page_pattern.search(all_text)
            if page_content:
                 final_context_docs = [page_content.group(1).strip()]
            else:
                final_context_docs = [f"ไม่พบเนื้อหาสำหรับหน้าที่ {page_num}"]
        else:
            print("📚 Summarization requested for the whole document.")
            final_context_docs = documents

        prompt_template = "Please provide a concise summary of the following text in Thai:\n\nContext:\n{context}"

    elif intent == "specific_question":
        print("🚀 Routing to: Specific Q&A Tool")
        top_rerank = 1
        
        query_vector = np.array(embed_queries([user_query]))
        if query_vector.size == 0:
            return {"error": "Could not generate query embedding."}

        faiss_dist, faiss_indices = faiss_index.search(query_vector, top_k=min(top_k, len(documents)))
        faiss_ranked = faiss_indices[0].tolist()
        
        rerank_scores = compute_scores(user_query, [documents[i] for i in faiss_ranked])
        reranked_results = sorted(zip(rerank_scores, faiss_ranked), reverse=True, key=lambda x: x[0])[:top_rerank]
        
        final_context_docs = [documents[i] for _, i in reranked_results]
        prompt_template = "Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    final_context = "\n\n".join(final_context_docs)
    
    if intent == "summarization":
        prompt = prompt_template.format(context=final_context)
        system_message = "You are a helpful assistant that summarizes text concisely in Thai, based ONLY on the provided context."
    else: # specific_question
        prompt = prompt_template.format(context=final_context, query=user_query)
        system_message = """You are an expert Q&A assistant. Your task is to answer the user's question based strictly and ONLY on the provided 'Context'.
- If the answer is found in the Context, provide the answer directly in Thai.
- If the answer is NOT in the Context, explain the system's limitation and provide better examples of specific questions. For example: 'ขออภัยค่ะ ระบบถูกออกแบบมาเพื่อตอบคำถามที่เฉพาะเจาะจงจากรายละเอียดในเอกสาร แทนที่จะถามว่า 'เอกสารเกี่ยวกับอะไร' กรุณาลองถามเจาะจงลงไปในเนื้อหา เช่น 'ในเอกสารกล่าวถึงสถานที่ใดบ้าง' ค่ะ'"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    
    image_tags = re.findall(r'!\[.*?\]\(.*?\)', final_context)
    
    final_response = answer
    if image_tags:
        final_response += "\n\n" + "\n\n".join(image_tags)
        print(f"✅ Appending {len(image_tags)} image tag(s) to the final response.")

    return {"generated_answer": final_response}

@app.get("/get-presigned-url/{image_id}")
async def get_presigned_url(image_id: str, username: str = Query(...), project_id: str = Query(...)):
    try:
        url = s3_storage.presigned_url(
            image_id=image_id,
            username=username,
            project_id=project_id,
            expires=3600
        )
        return JSONResponse(content={"url": url})
    except Exception as e:
        print(f"Error generating presigned URL for {image_id}: {e}")
        return JSONResponse(status_code=404, content={"error": "Image not found or error generating URL"})

@app.get("/status")
async def status():
    return {
        "faiss_ready": faiss_index.index.ntotal > 0,
        "bm25_ready": bm25 is not None,
        "documents_loaded": bool(documents)
    }