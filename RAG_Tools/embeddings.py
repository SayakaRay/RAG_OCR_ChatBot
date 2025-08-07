# RAG_Tools/embeddings.py (เวอร์ชันใช้ SentenceTransformer)

from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# --- จุดที่ 1: เปลี่ยนการโหลดโมเดลมาใช้ SentenceTransformer ---

print("[Embedding Model] Initializing BAAI/bge-m3 model with SentenceTransformer...")

# ตรวจสอบว่ามี GPU หรือไม่ และเลือก device (แต่จะบังคับใช้ CPU ตามคำขอก่อนหน้า)
device = "cuda" if torch.cuda.is_available() else "cpu"
# บังคับใช้ CPU เพื่อแก้ปัญหาความเข้ากันได้ของการ์ดจอ
device = "cpu" 
print(f"[Embedding Model] Forcing device to use: {device}")

# โหลดโมเดลและส่งไปที่ device ที่ต้องการ
model = SentenceTransformer("BAAI/bge-m3", device=device)

print("[Embedding Model] ✅ Model initialized.")


def embed_text(text: str):
    """สร้าง Embedding สำหรับข้อความเดียว"""
    if not isinstance(text, str) or not text.strip():
        print("⚠️ [Embedding] embed_text received an invalid input. Skipping.")
        return None
        
    # --- จุดที่ 2: เปลี่ยนวิธีการเรียกใช้ model.encode ---
    # .encode() ของ SentenceTransformer จะคืนค่าเป็น numpy array โดยตรง
    embedding = model.encode([text])
    return embedding[0] if len(embedding) > 0 else None

def embed_queries(texts: list[str]):
    """สร้าง Embeddings สำหรับ Queries (List ของข้อความ)"""
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not valid_texts:
        print("⚠️ [Embedding] No valid queries to embed after filtering.")
        return np.array([])

    return model.encode(valid_texts)

def embed_texts(texts: list[str]):
    """
    สร้าง Embeddings สำหรับ Documents (List ของข้อความ)
    """
    if not texts:
        return np.array([])
    
    print(f"[Embedding] Received {len(texts)} chunks for embedding.")
    
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    
    if len(valid_texts) < len(texts):
        print(f"⚠️ [Embedding] Filtered out {len(texts) - len(valid_texts)} invalid or empty chunks.")

    if not valid_texts:
        print("❌ [Embedding] No valid texts to embed after filtering.")
        return np.array([])

    print(f"[Embedding] Sending {len(valid_texts)} valid chunks to the model...")
    embeddings = model.encode(valid_texts, batch_size=16) # batch_size ช่วยให้ประมวลผลเร็วขึ้น
    print("[Embedding] ✅ Embedding complete.")
    return embeddings

# --- (ส่วน if __name__ == "__main__" สำหรับทดสอบ) ---
if __name__ == "__main__":
    queries = ["What is the capital of China?", "Explain gravity"]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts bodies.",
        "", 
        "   ",
        None
    ]

    print("\n--- Testing embed_queries ---")
    q_emb = embed_queries(queries)
    print("Query embeddings shape:", q_emb.shape)
    
    print("\n--- Testing embed_texts ---")
    d_emb = embed_texts(documents)
    print("Document embeddings shape:", d_emb.shape)

    if d_emb is not None and d_emb.size > 0:
        sim = q_emb @ d_emb.T
        print("\nSimilarity matrix:\n", sim)