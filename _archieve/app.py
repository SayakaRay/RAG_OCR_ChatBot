from fastapi import FastAPI, Query
from embeddings import embed_texts,embed_queries,embed_text
from faiss_index import FaissIndex
from text import documents
from openai import OpenAI
from dotenv import load_dotenv

from reranker import compute_scores
from chunk_text import recursive_chunk
import os

load_dotenv() #Load OpenAI Key

client = OpenAI() #Create OpenAI
app = FastAPI() #Create FastAPI

@app.get("/")
async def root():
    return {"message": "Hello World"}

# #Embed existing documents
# print("Embedding existing documents")
# doc_embeddings = embed_texts(documents)
# dimension = doc_embeddings.shape[1]

# #Build FAISS index
# print("FAISS index built.")
# faiss_index = FaissIndex(dimension=dimension)
# faiss_index.build_index(doc_embeddings)

# Load and chunk document
with open("./text_document/thai_paper.txt", "r", encoding="utf-8") as f:
    big_text = f.read()

# Embed existing documents
documents = recursive_chunk(big_text, max_words=256, overlap_words=60)
print(f"Loaded and chunked {len(documents)} documents.")
doc_embeddings = embed_texts(documents)

# Build FAISS index
print("FAISS index built.")
faiss_index = FaissIndex()
faiss_index.build_index(doc_embeddings)

@app.get("/retrieval")
async def retrieval(user_query: str, top_k: int = 1):
    # Embed query
    query_embedding = embed_queries([user_query])

    # Search FAISS
    distances, indices = faiss_index.search(query_embedding, top_k=top_k)

    results = []
    for rank, num in enumerate(indices[0], start=1):
        print("IDK : "+str(num))
        results.append({
            "rank": rank,
            "document": documents[num],
            "distance": float(distances[0][rank-1])
        })

    return {
        "query": user_query,
        "top_k": top_k,
        "matches": results
    }

@app.get("/generate")
async def generate(user_query: str, top_k: int = 3):
    # Embed query
    query_embedding = embed_queries([user_query])

    # Search FAISS
    distances, indices = faiss_index.search(query_embedding, top_k=top_k)
    
    # Get retrieved documents from FAISS
    retrieved_data = [documents[num] for num in indices[0]]

    # Use reranker to score them
    scores = compute_scores(user_query, retrieved_data)

    # Sort by scores
    scored_docs = sorted(zip(retrieved_data, scores), key=lambda x: x[1], reverse=True)

    # Take top reranked documents
    reranked_top_docs = [doc for doc, score in scored_docs[:top_k]]
    context = "\n".join(reranked_top_docs)


    # Prompt generation
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    print(prompt)
    # Call OpenAI
    try:
        response = client.responses.create(
            model="gpt-4o-mini-2024-07-18",
            input=[
                {"role": "system", "content": "You are a helpful assistant. Only use the context below to answer the question , DON'T retrieved data from other source. Also, make the pun with an answer if possible ,then end the joke with an emoji"},
                {"role": "user", "content": prompt}
            ],
        )
        answer = response.output_text.strip()

    except Exception as e:
        return {"error": str(e)}

    # Return answer
    return {
        "query": user_query,
        "top_k": top_k,
        "matches": [
            {"document": doc, "score": score}
            for doc, score in scored_docs[:top_k]
        ],
        "generated_answer": answer
    }