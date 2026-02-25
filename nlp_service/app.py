# Servicio de procesamiento de lenguaje natural (NLP) especializado

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import torch
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
import io
import gc
import re  # Necesario para las expresiones regulares
import requests
import json

# Importar procesadores NLP
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NLP_Service")

class WhisperManager:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self, device=None):
        if self.model is not None:
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Determinar dispositivo
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validar si pide cuda pero no hay
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("Solicitado CUDA pero no disponible. Usando CPU.")
            device = "cpu"

        compute = "float16" if device == "cuda" else "int8"
        logger.info(f"Cargando Whisper (NLP) en: {device.upper()}")
        
        self.model = WhisperModel(
            "medium",
            device=device,
            compute_type=compute
        )

    async def transcribe_bytes(self, audio_bytes: bytes, language="es"):
        with io.BytesIO(audio_bytes) as bio:
            audio, _ = sf.read(bio, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

        segments, info = self.model.transcribe(
            audio,
            language=language,
            vad_filter=True
        )

        return " ".join(seg.text for seg in segments)

whisper_manager = WhisperManager()

# Inicializar aplicación FastAPI
app = FastAPI(
    title="NLP Service - TFM Sistema de Recomendación",
    description="Servicio especializado en procesamiento de lenguaje natural para análisis de preferencias",
    version="1.0.0"
)

# Modelos de IA cargados
nlp_models = {
    'sentiment_analyzer': None,
    'spacy_processor': None,
    'sentence_encoder': None
}

# --- LÓGICA DE ALTA PRECISIÓN ---

""" [LEGACY - REGLAS] Comentado para usar arquitectura Agéntica
def detect_reference_movie(text: str) -> Optional[str]:
    ""Detecta 'películas tipo X' o 'como X'.""
    patterns = [
        r"tipo\s+(.+?)(?:$|\.|,)",
        r"como\s+(.+?)(?:$|\.|,)",
        r"parecidas\s+a\s+(.+?)(?:$|\.|,)",
        r"similares\s+a\s+(.+?)(?:$|\.|,)",
        r"estilo\s+(.+?)(?:$|\.|,)"
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) > 2 and candidate.lower() not in ["las", "una", "unas", "el", "películas"]:
                return candidate
    return None

def detect_sort_intent(text: str) -> str:
    ""Detecta si el usuario quiere 'mejores', 'nuevas', etc.""
    t = text.lower()
    if "mejor" in t or "top" in t or "maestras" in t or "buenas" in t:
        return "vote_average.desc"
    if "peor" in t or "malas" in t:
        return "vote_average.asc"
    if "nueva" in t or "reciente" in t or "últim" in t or "estreno" in t:
        return "primary_release_date.desc"
    if "antigua" in t or "vieja" in t or "clásic" in t:
        return "primary_release_date.asc"
    return "popularity.desc" # Default
"""

# --- ENDPOINT DE CONTROL DE DISPOSITIVO ---
class DeviceConfig(BaseModel):
    device: str

@app.post("/system/set-device")
async def set_device(config: DeviceConfig):
    """Recibe orden de cambio de dispositivo y recarga Whisper Y SBERT."""
    try:
        # 1. Recargar Whisper
        whisper_manager.load_model(device=config.device)

        # 2. Recargar Sentence Transformer (Para que el ranking semántico use GPU)
        """ [LEGACY] Ya no cargamos modelos locales pesados en NLP service
        if nlp_models['sentence_encoder']:
             logger.info(f"🔄 Recargando SentenceTransformer en {config.device.upper()}...")
             nlp_models['sentence_encoder'] = SentenceTransformer(
                'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                device=config.device
            )
        """

        return {"status": "ok", "device": config.device}
    except Exception as e:
        logger.error(f"Error cambiando dispositivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class PreferenceExtractionRequest(BaseModel):
    """Modelo para extracción de preferencias."""
    text: str
    conversation_context: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Cargar modelos de NLP al iniciar el servicio."""
    logger.info("🚀 Iniciando servicio NLP...")
    
    try:
        # Cargar modelos NLP
        await load_nlp_models()
        logger.info("✅ Todos los modelos NLP cargados exitosamente")
        
        logger.info("✅ Servicio NLP listo (Conectado a Ollama)")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos NLP: {e}")
        raise

async def load_nlp_models():
    """Cargar todos los modelos de NLP necesarios."""
    logger.info("🧠 Cargando modelos de NLP...")
    
    try:
        # Usamos GPU si está disponible por defecto al inicio
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Procesador SpaCy para español
        """ [LEGACY] Comentado: Delegamos el entendimiento al LLM
        logger.info("🔍 Cargando procesador SpaCy...")
        nlp_models['spacy_processor'] = spacy.load("es_core_news_sm")
        
        # Encoder de sentencias para similitud semántica
        logger.info("🔤 Cargando encoder de sentencias...")
        nlp_models['sentence_encoder'] = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        """
        
        logger.info("✅ Modelos NLP cargados")
        
    except Exception as e:
        logger.error(f"❌ Error cargando modelos: {e}")
        raise

@app.get("/health")
async def health_check():
    """Verificar estado del servicio y modelos."""
    models_status = {
        name: model is not None 
        for name, model in nlp_models.items()
    }
    # Verificar conexión con Ollama
    ollama_status = False
    try:
        r = requests.get("http://ollama:11434/api/tags", timeout=2)
        ollama_status = r.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy" if all(models_status.values()) else "degraded",
        "models": models_status,
        "status": "healthy" if ollama_status else "degraded",
        "ollama_connected": ollama_status,
        "timestamp": datetime.now().isoformat()
    }

def query_ollama_agent(text: str, context: dict) -> dict:
    """Consulta al LLM para extraer estructura JSON."""
    
    system_prompt = """
    Eres un experto agente de recomendación de películas. Tu trabajo es analizar el texto del usuario y extraer parámetros de búsqueda estructurados en formato JSON.
    
    Reglas de extracción:
    1. 'intent': Puede ser 'recommend_movie', 'greeting', 'exit' o 'other'.
    2. 'genres': Lista de géneros en INGLÉS (ej: 'Action', 'Horror', 'Sci-Fi'). Traduce del español.
    3. 'years': Lista de años o décadas (ej: '1980s', '2023').
    4. 'limit': Número entero de películas solicitadas (default 5).
    5. 'sort_by': 'popularity.desc' (default), 'vote_average.desc' (si pide mejores/buenas), 'primary_release_date.desc' (si pide nuevas).
    6. 'detected_name': Si el usuario se presenta ("Soy Juan"), extrae el nombre. Si no, null.
    7. 'keywords': Palabras clave importantes para la trama (ej: "espacio", "zombies").
    
    Responde ÚNICAMENTE con el JSON válido.
    """

    payload = {
        "model": "llama3", # Asegúrate de tener este modelo o cambia a 'mistral'
        "format": "json",
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Contexto previo: {context}\n\nUsuario dice: {text}"}
        ]
    }

    try:
        response = requests.post("http://ollama:11434/api/chat", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        content = result['message']['content']
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error Ollama: {e}")
        # Fallback básico si falla el LLM
        return {"intent": "recommend_movie", "genres": [], "keywords": [text], "limit": 5}

@app.post("/extract-preferences")
async def extract_preferences(request: PreferenceExtractionRequest):
    """
    Extracción avanzada de preferencias cinematográficas del texto.
    Extracción Agéntica de preferencias usando LLM.
    """
    try:
        text = request.text
        text_lower = text.lower()
        context = request.conversation_context or {}
        
        # --- NUEVA ARQUITECTURA AGÉNTICA (LLM) ---
        # Llamada al Agente LLM
        llm_response = query_ollama_agent(text, context)
        
        # Normalizar respuesta para el backend
        preferences = {
            "genres": llm_response.get("genres", []),
            "years": llm_response.get("years", []),
            "keywords": llm_response.get("keywords", []),
            "actors": [], # El LLM podría extraer esto también si ajustamos el prompt
            "sort_by": llm_response.get("sort_by", "popularity.desc"),
            "limit": llm_response.get("limit", 5),
            "detected_name": llm_response.get("detected_name"),
            "intent": llm_response.get("intent", "recommend_movie"),
            "raw_llm": llm_response,
            "conversation_context": request.conversation_context or {}
        }

        """ [LEGACY] Lógica basada en reglas (Regex/Spacy) - Desactivada
        # 1. Detección de Cantidad de películas pedidas
        number_map = {"una":1,"uno":1,"dos":2,"tres":3,"cuatro":4,"cinco":5,"diez":10}
        digit_match = re.search(r'\b(\d+)\b', text_lower)
        return {"preferences": preferences, "intent": preferences["intent"]}
        
        if digit_match:
            preferences["limit"] = int(digit_match.group(1))
        else:
            for w, n in number_map.items():
                if f" {w} " in f" {text_lower} ": 
                    preferences["limit"] = n; break
        preferences["limit"] = min(max(preferences["limit"], 1), 20)

        # 2. Detección de Referencia y Ordenación
        preferences["reference_movie"] = detect_reference_movie(text)
        preferences["sort_by"] = detect_sort_intent(text)

        # 3. Detección Clásica (Géneros, Años)
        preferences["genres"] = detect_movie_genres(text_lower)
        preferences["years"] = detect_years_periods(text_lower)
        
        # 4. Keywords y Entidades (Spacy)
        if nlp_models['spacy_processor']:
            doc = nlp_models['spacy_processor'](text)
            kws = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 2]
            preferences["keywords"] = kws[:6]
            preferences["actors"] = [e.text for e in doc.ents if e.label_ == "PER"]
            
        # 5. Definir Intención para el Backend
        if preferences["reference_movie"]:
            preferences["search_intent"] = "similar_to"
        elif preferences["genres"] or preferences["keywords"]:
            preferences["search_intent"] = "filtered"
        else:
            preferences["search_intent"] = "semantic_exploration"
        """
            
        return {"preferences": preferences, "intent": preferences["intent"]}
        
    except Exception as e:
        logger.error(f"❌ Error extrayendo preferencias: {e}")
        # Retornar estructura vacía segura en caso de error
        return {"preferences": {}}

# FUNCIONES DE DETECCIÓN ESPECÍFICAS
""" [LEGACY] Funciones auxiliares de reglas
def detect_movie_genres(text: str) -> List[str]:
    ""Detectar géneros cinematográficos en el texto.""
    mapping = {
        'acción': 'Action',
        'comedia': 'Comedy', 
        'drama': 'Drama',
        'terror': 'Horror',
        'romance': 'Romance',
        'ciencia ficción': 'Science Fiction',
        'aventura': 'Adventure',
        'animación': 'Animation',
        'thriller': 'Thriller',
        'fantasía': 'Fantasy',
        'crimen': 'Crime',
        'misterio': 'Mystery',
        'suspense': 'Thriller',
        'documental': 'Documentary',
        'familia': 'Family',
        'guerra': 'War',
        'historia': 'History',
        'musical': 'Music',
        'western': 'Western'
    }
    
    return [en for es, en in mapping.items() if es in text]

def detect_years_periods(text: str) -> List[str]:
    ""Detectar años o períodos temporales mencionados.""
    
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    if "80" in text: years.append("1980s")
    if "90" in text: years.append("1990s")
    if "antigua" in text or "clásica" in text: years.append("antiguas")
    return years

def detect_moods(text: str) -> List[str]:
    ""Detectar estados de ánimo o tonos deseados.""
    mood_keywords = {
        'divertida': 'funny',
        'emocionante': 'exciting', 
        'triste': 'sad',
        'romántica': 'romantic',
        'aterradora': 'scary',
        'suspense': 'suspenseful',
        'inspiradora': 'inspiring',
        'oscura': 'dark',
        'ligera': 'light',
        'intensa': 'intense'
    }
    
    detected_moods = []
    for spanish_mood, english_mood in mood_keywords.items():
        if spanish_mood in text:
            detected_moods.append(english_mood)
    
    return detected_moods
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)