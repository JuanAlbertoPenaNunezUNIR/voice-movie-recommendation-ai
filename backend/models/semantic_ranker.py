# backend/models/semantic_ranker.py

from sentence_transformers import SentenceTransformer, util
import torch

class SemanticRanker:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=self.device
        )

    def rank(self, query: str, movies: list, top_k=10):
        if not movies:
            return []

        texts = [
            movie.get("overview", "") for movie in movies
        ]

        query_emb = self.model.encode(query, convert_to_tensor=True)
        docs_emb = self.model.encode(texts, convert_to_tensor=True)

        scores = util.cos_sim(query_emb, docs_emb)[0]

        ranked = sorted(
            zip(movies, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [m for m, _ in ranked[:top_k]]