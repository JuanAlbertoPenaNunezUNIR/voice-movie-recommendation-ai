# Procesador de lenguaje natural avanzado para análisis conversacional profundo
# Cliente NLP que conecta con el Agente LLM (nlp_service)

import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import asyncio
from utils.device_manager import device_manager  # Usamos el gestor centralizado
import os
import requests
import logging

logger = logging.getLogger(__name__)

class NLPProcessor:

    def __init__(self):
        self.model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.tokenizer = None
        self.model = None
        
        # Mapeo de géneros en español a términos estándar (inglés o IDs para TMDB)
        self.genre_map = {
            "ciencia ficción": "science fiction", "sci-fi": "science fiction", "futurista": "science fiction",
            "terror": "horror", "miedo": "horror", "susto": "horror",
            "acción": "action", "aventura": "adventure",
            "comedia": "comedy", "risa": "comedy", "divertida": "comedy",
            "drama": "drama", "romántica": "romance", "amor": "romance",
            "animación": "animation", "dibujos": "animation",
            "fantasía": "fantasy", "documental": "documentary",
            "suspense": "thriller", "intriga": "thriller", "policiaca": "crime", "crimen": "crime"
        }
        # URL del servicio NLP (definida en docker-compose o default)
        self.nlp_service_url = os.getenv("NLP_SERVICE_URL", "http://nlp_service:8001")

        self.intents = ["recommend_movie", "greeting", "exit", "other"]

    async def initialize(self):
        """Inicializa o mueve el modelo según el dispositivo seleccionado en el frontend."""
        target_device = device_manager.get_device_str()
        # La lógica de carga local se ha delegado completamente al servicio NLP o al Agente LLM.
        pass

    def classify_intent(self, text: str) -> str:
        """ [LEGACY] Clasificación local desactivada """
        # Usamos device_manager para saber dónde poner los tensores de entrada
        # device = device_manager.get_device_str()
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(device)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     idx = torch.argmax(outputs.logits, dim=1).item()
        # return self.intents[idx]
        return "other"
    
    def extract_name(self, text: str) -> str:
        """[LEGACY] Extracción local desactivada - Delegada al Agente"""
        return None

    def extract_entities(self, text: str) -> dict:
        """[LEGACY] Extracción de entidades local desactivada"""
        return {}

    async def process(self, text: str):
        """
        Delega el procesamiento al Agente LLM en nlp_service.
        """
        
        try:
            # Llamada al servicio NLP (que ahora usa Ollama)
            response = requests.post(
                f"{self.nlp_service_url}/extract-preferences",
                json={"text": text, "conversation_context": {}},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                preferences = data.get("preferences", {})
                
                return {
                    "intent": data.get("intent", "other"),
                    "entities": preferences,
                    "detected_name": preferences.get("detected_name"),
                    "raw_text": text
                }
            else:
                logger.error(f"Error NLP Service: {response.status_code}")
                return {"intent": "other", "entities": {}, "raw_text": text}
                
        except Exception as e:
            logger.error(f"Excepción conectando con NLP Service: {e}")
            return {"intent": "other", "entities": {}, "raw_text": text}
