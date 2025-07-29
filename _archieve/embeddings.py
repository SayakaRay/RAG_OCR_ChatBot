from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def embed_text(text: str):
    embeddings = model.encode([text])
    return embeddings[0]

def embed_queries(texts: list[str]):
    return model.encode(texts, prompt_name="query", batch_size=16, show_progress_bar=True)

def embed_texts(texts: list[str]):
    return model.encode(texts, batch_size=16, show_progress_bar=True)

# Testing enbedding system
if __name__ == "__main__":
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    query_embeddings = embed_queries(queries)
    document_embeddings = embed_texts(documents)

    similarity = model.similarity(query_embeddings, document_embeddings)
    print("Similarity matrix:")
    print(similarity)
