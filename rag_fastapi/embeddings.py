from sentence_transformers import SentenceTransformer

# Choose a small, fast model for local use
EMBED_MODEL = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed(self, texts):
        # returns list of vectors
        return self.model.encode(texts, show_progress_bar=False).tolist()
