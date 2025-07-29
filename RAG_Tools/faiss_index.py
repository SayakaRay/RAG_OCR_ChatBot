import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
    
    def build_index(self, embeddings: np.ndarray): #Add embeddings to the FAISS index.
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        print(f"Index built : {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k=3): #Search for the top_k most similar vectors to the query.
        query_embedding = query_embedding.astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices


if __name__ == "__main__":
    dim = 1024 # number of dimension
    nb = 1000  # number of database vectors
    nq = 5     # number of queries
    top_k = 4  # number of neighbors to retrieve

    np.random.seed(123)
    database_vectors = np.random.rand(nb, dim).astype('float32')
    query_vectors = np.random.rand(nq, dim).astype('float32')

    # Initialize FAISS index
    faiss_index = FaissIndex(dimension=dim)

    # Build index with database vectors
    faiss_index.build_index(database_vectors)

    # Search using the first 5 database vectors themselves as queries
    distances, indices = faiss_index.search(database_vectors[:nq], top_k=top_k)
    print("\nSanity Check - Searching with vectors from the index itself:")
    print("Indices (should show nearest neighbors, ideally themselves):")
    print(indices)
    print("Distances:")
    print(distances)

    # Actual search with random query vectors
    distances, indices = faiss_index.search(query_vectors, top_k=top_k)
    print("\nSearch Results for random query vectors:")
    print("Indices:")
    print(indices)
    print("Distances:")
    print(distances)
