from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned(
    "BAAI/bge-m3",
    query_instruction_for_retrieval="",
    use_fp16=True
)

def embed_text(text: str):
    dense = model.encode([text], batch_size=16)["dense_vecs"][0]
    return dense

def embed_queries(texts: list[str]):
    return model.encode(texts, batch_size=16)["dense_vecs"]

def embed_texts(texts: list[str]):
    return model.encode(texts, batch_size=16)["dense_vecs"]

if __name__ == "__main__":
    queries = ["What is the capital of China?", "Explain gravity"]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts bodies."
    ]

    q_emb = embed_queries(queries)
    d_emb = embed_texts(documents)

    sim = q_emb @ d_emb.T
    print("Similarity matrix:\n", sim)
