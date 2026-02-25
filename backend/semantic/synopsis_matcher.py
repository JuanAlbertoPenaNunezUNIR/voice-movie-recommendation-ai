from sentence_transformers import SentenceTransformer
import numpy as np
from utils.device_manager import device_manager
import logging

class SynopsisMatcher:
    """
    Calcula similitud semántica entre el prompt del usuario
    y las sinopsis de las películas (SBERT).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def initialize(self):
        """
        Inicializa o recarga el modelo SBERT en el dispositivo seleccionado.
        Se puede llamar múltiples veces (CPU <-> GPU).
        """
        device = device_manager.get_device_str()

        self.logger.info(f"🔄 Cargando SBERT ({self.model_name}) en {device.upper()}")

        self.model = SentenceTransformer(
            self.model_name,
            device=device
        )

    async def calculate_scores(self, query: str, movies: list) -> list:
        """
        Añade _semantic_score a cada película basado en similitud
        entre la query y la sinopsis.
        """

        if not self.model:
            # Seguridad extra: inicializar si alguien olvidó hacerlo
            self.initialize()

        # Embedding del query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        for movie in movies:
            synopsis = movie.get("overview") or ""
            if not synopsis.strip():
                movie["_semantic_score"] = 0.0
                continue

            movie_embedding = self.model.encode(
                synopsis,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            # Cosine similarity (producto escalar al estar normalizados)
            score = float(np.dot(query_embedding, movie_embedding))
            movie["_semantic_score"] = score

        return movies