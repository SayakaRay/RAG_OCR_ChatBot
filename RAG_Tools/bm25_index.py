import re
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize as thai_word_tokenize

def hybrid_tokenize(text: str) -> list[str]:
    """Tokenize Thai + English text using PyThaiNLP + regex fallback."""
    thai_tokens = thai_word_tokenize(text, engine="newmm")
    split_tokens = []
    for tok in thai_tokens:
        split = re.findall(r"\w+|[^\w\s]", tok, re.UNICODE)
        split_tokens.extend([s for s in split if s.strip()])
    return split_tokens

class BM25Retriever:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized_docs = [hybrid_tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_scores(self, query: str) -> list[float]:
        """Return BM25 scores for all documents."""
        q_tokens = hybrid_tokenize(query)
        return self.bm25.get_scores(q_tokens)

    def get_top_n(self, query: str, n: int = 5) -> list[str]:
        """Return top N documents (text only)."""
        q_tokens = hybrid_tokenize(query)
        return self.bm25.get_top_n(q_tokens, self.documents, n=n)

    def get_top_indices_with_scores(self, query: str, n: int = 5) -> list[tuple[int, float]]:
        """Return top N (index, score) pairs for merging with other retrievers."""
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
        return ranked

# Optional CLI test
if __name__ == "__main__":
    documents = [
        "FastAPI is a modern, fast web framework for building APIs.",
        "ฉันชอบไปเที่ยวเชียงใหม่ในฤดูหนาว",
        "FAISS is great for vector search.",
        "BM25 is for keyword ranking.",
        "วันนี้อากาศดีมากที่กรุงเทพฯ",
        "I love eating Pad Thai in Bangkok.",
        "Python is widely used for data science.",
        "ภาษาไทยมีความสวยงามและซับซ้อน",
        "Let's travel to Chiang Mai in winter!",
        "การค้นหาข้อมูลด้วยคีย์เวิร์ดมีประโยชน์มาก",
    ]

    retriever = BM25Retriever(documents)

    test_queries = [
        "ไปเที่ยวเชียงใหม่",
        "ไปเที่ยวเชียงใหม่ในฤดูหนาว",
        "data science with Python",
        "Pad Thai กรุงเทพฯ"
    ]
    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        results = retriever.get_top_indices_with_scores(query, n=3)
        for idx, score in results:
            print(f"→ {score:.4f} | {documents[idx]}")
