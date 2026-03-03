from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
import tempfile
import os
import uuid
import glob
import logging
import shutil
import torch
from pydub import AudioSegment
import time
import platform
import io

import redis.asyncio as redis
import requests

# Importar agente LLM
from services.llm_agent import run_llm_agent, generate_recommendation_response

# Importar servicios propios
from services.tmdb_service import TMDBService
from services.recommendation_service import RecommendationService
from models.recommendation_model import RecommendationEngine
from models.nlp_processor import NLPProcessor
from utils.device_manager import device_manager

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#Configuración
VOICE_DIR = "/app/voice_clones"
os.makedirs(VOICE_DIR, exist_ok=True)

# Configuración de servicios externos
NLP_SERVICE_URL = os.getenv("NLP_SERVICE_URL", "http://nlp_service:8001")
redis_client = None
nlp_processor = NLPProcessor() # Instancia local del procesador NLP

# Inicialización de servicios
tmdb_service = TMDBService()
rec_engine = RecommendationEngine() 
rec_service = RecommendationService(rec_engine, tmdb_service)

# --- GESTIÓN DE MODELOS Y HARDWARE ---
async def load_models_async():
    """Recarga modelos locales y notifica al NLP Service."""
    device = device_manager.get_device_str()
    
    logger.info(f"🔄 RECARGANDO TODO EL SISTEMA EN: {device.upper()}...")
    device_manager.clear_cache()
    
    # 1. Notificar a NLP Service para que mueva sus modelos
    try:
        requests.post(f"{NLP_SERVICE_URL}/system/set-device", json={"device": device}, timeout=5)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo sincronizar dispositivo con NLP Service: {e}")

    # 2. Recargar NLP Processor Local (si queda algo)
    try:
        await nlp_processor.initialize() 
        logger.info(f"✅ NLP Processor movido a {device}")
    except Exception as e:
        logger.error(f"❌ Error NLP: {e}")

    # 3. Recargar Motor Semántico (Similitud de Sinopsis)
    try:
        rec_service.reload_model()
        logger.info(f"✅ Motor Semántico movido a {device}")
    except Exception as e:
        logger.error(f"❌ Error Motor Semántico: {e}")
        
    device_manager.clear_cache()
    logger.info("🚀 SISTEMA COMPLETAMENTE OPERATIVO")

# ============================
# ENDPOINTS DE SISTEMA
# ============================

@app.on_event("startup")
async def startup():
    global redis_client
    # Conexión a Redis
    try:
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
        redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await redis_client.ping()
        logger.info("✅ Conectado a Redis correctamente")
        tmdb_service.set_cache(redis_client)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo conectar a Redis (Caché desactivado): {e}")
        
    await tmdb_service.initialize()
    await load_models_async()

@app.on_event("shutdown")
async def shutdown_event():
    """Cerrar conexiones al detener el servicio."""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("🔒 Conexión Redis cerrada")

@app.post("/system/set-device")
async def set_system_device(payload: dict, background_tasks: BackgroundTasks):
    """Cambia el dispositivo globalmente y recarga todo."""
    target_device = payload.get("device")
    try:
        changed = device_manager.set_device(target_device)
        if changed:
            # Lanzamos la recarga en background para no bloquear al usuario
            background_tasks.add_task(load_models_async)
            return {"status": "ok", "message": f"Cambiando sistema a {target_device.upper()}. Recargando modelos..."}
        return {"status": "ok", "message": f"El sistema ya está en {target_device.upper()}."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/system/status")
def get_system_status():
    return device_manager.get_status()

@app.get("/system/metrics")
def get_metrics():
    """Métricas del sistema (Hardware real + Métricas IA simuladas)."""
    import psutil
    
    def get_cpu_name():
        try:
            if platform.system() == "Linux":
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            return platform.processor()
        except: return "CPU Genérica"
    
    device_status = device_manager.get_status()
    is_cuda = device_status["current_device"] == "cuda"
    
    mem = psutil.virtual_memory()
    
    metrics = {
        "hardware": {
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage_gb": round(mem.used / (1024**3), 2),
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "device_mode": "GPU Nvidia - CUDA" if is_cuda else "CPU",
            "redis_status": "Connected" if redis_client else "Disconnected",
            "gpu_name": device_status.get("gpu_name", "N/A"),
            "cpu_name": get_cpu_name()
        },
        "whisper": {
            "wer": 0.082, 
            "avg_latency_ms": 450 if is_cuda else 2100,
            "precision_cinematographic": "High (Ontology Based)"
        },
        "recommendation": {
            "precision_at_10": 0.81,
            "recall_at_10": 0.74,
            "ndcg_at_10": 0.86,
            "semantic_relevance_avg": 0.72,
            "task_success_rate": 0.83,
            "avg_turns_to_success": 1.4
        },
        "system": {
            "total_latency_avg_sec": 3.5 if is_cuda else 12.0,
            "availability": "99.9%"
        }
    }
    
    if is_cuda:
        try:
            metrics["hardware"]["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        except: 
            pass
            
    return metrics

# ============================
# GESTIÓN DE VOCES
# ============================
def process_audio_for_cloning(input_path: str, output_path: str):
    """Convierte audio a WAV Mono 24kHz."""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(24000).set_channels(1)
        audio.export(output_path, format="wav")
    except Exception as e:
        logger.error(f"Error procesando audio: {e}")
        shutil.copy(input_path, output_path)

@app.get("/list-voices")
def list_voices():
    files = glob.glob(os.path.join(VOICE_DIR, "*.wav"))
    voices = [os.path.basename(f) for f in files]
    if not voices:
        return ["default (sistema)"]
    return sorted(voices)

@app.post("/clone-voice")
async def clone_voice(file: UploadFile = File(...), name: str = Form(...)):
    try:
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip() or "voz_custom"
        filename = f"{safe_name}.wav"
        file_path = os.path.join(VOICE_DIR, filename)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        process_audio_for_cloning(tmp_path, file_path)
        os.remove(tmp_path)
        
        return {"status": "ok", "message": f"Voz '{safe_name}' creada.", "filename": filename}
    except Exception as e:
        logger.error(f"Error clonando: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-voice")
def delete_voice(id: str):
    """Elimina una voz clonada por su nombre de archivo."""
    # Sanitizar nombre de archivo para seguridad
    safe_filename = os.path.basename(id)
    file_path = os.path.join(VOICE_DIR, safe_filename)
    
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"status": "ok", "message": f"Voz {safe_filename} eliminada."}
    
    # Si no existe, devolvemos OK igualmente para que el frontend se actualice y la elimine de la lista visual
    return {"status": "ok", "message": "Voz no encontrada, se asume eliminada."}

# ============================
# STT & TTS
# ============================
@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    if whisper_model is None:
         return {"text": "El modelo se está cargando, espera un momento..."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Transcribir
        segments, _ = whisper_model.transcribe(tmp_path, language="es", beam_size=5)
        
        # Consumir generador con seguridad para evitar crash en iteración
        text_segments = []
        for seg in segments:
            text_segments.append(seg.text)
            
        text = " ".join(text_segments).strip()
        return {"text": text}
    except Exception as e:
        logger.error(f"❌ Error STT Crítico: {e}")
        return {"text": "", "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/text-to-speech")
async def text_to_speech(payload: dict):
    text = payload.get("text", "Hola")
    voice_file = payload.get("voice", "")
    
    # Limpiar nombre de voz para enviar solo el filename
    voice_clean = voice_file.replace("default (sistema)", "default")

    try:
        # Delegar síntesis al microservicio NLP
        resp = requests.post(
            f"{NLP_SERVICE_URL}/tts", 
            json={"text": text, "voice": voice_clean},
            timeout=60,
            stream=True
        )
        
        if resp.status_code != 200:
             raise HTTPException(status_code=500, detail=f"Error NLP TTS: {resp.text}")
             
        return StreamingResponse(io.BytesIO(resp.content), media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# LÓGICA PRINCIPAL: PROCESAMIENTO NLP
# ============================

def translate_genres_to_spanish(genres_list: list) -> list:
    mapping = {
        "action": "Acción", "adventure": "Aventura", "animation": "Animación",
        "comedy": "Comedia", "crime": "Crimen", "documentary": "Documental",
        "drama": "Drama", "family": "Familia", "fantasy": "Fantasía",
        "history": "Historia", "horror": "Terror", "music": "Música",
        "mystery": "Misterio", "romance": "Romance", "science fiction": "Ciencia Ficción",
        "tv movie": "Película de TV", "thriller": "Suspense", "war": "Guerra", "western": "Western"
    }
    # Normalizamos a minúsculas para buscar en el mapa
    return [mapping.get(g.lower(), g.capitalize()) for g in genres_list]

@app.post("/process-text")
async def process_text(payload: dict):
    user_text = payload.get("text", "")
    current_user_name = payload.get("user_name", "Usuario")

    # [OPTIMIZACIÓN] Desactivamos NLP Local para usar solo el Agente LLM
    # analysis_local = await nlp_processor.process(user_text)
    # detected_name = analysis_local.get("detected_name")
    #
    # logger.info(f"NLP ANALYSIS LOCAL => {analysis_local}")
    #
    # # --- Si se detecta un nombre, devolvemos saludo inmediato ---
    # if detected_name:
    #     return {
    #         "response": f"¡Hola, {detected_name}! ¡Pulsa el icono del micrófono para ampliar tu horizonte cinéfilo!",
    #         "recommendations": [],
    #         "detected_name": detected_name,
    #         "suggest_edit": False
    #     }

    # --- AGENTE LLM ---
    try:
        agent = run_llm_agent(user_text)
    except Exception as e:
        logger.error(f"LLM ERROR: {e}")
        return {
            "response": "He tenido un pequeño problema pensando 🤖 ¿Puedes repetirlo?",
            "recommendations": [],
            "suggest_edit": True,
            "transcription": user_text
        }

    intent = agent.get("intent")
    filters = agent.get("filters", {})
    response_text = agent.get("response", "")
    detected_name = agent.get("detected_name") # Extraer nombre si el agente lo detectó

    logger.info(f"LLM ANALYSIS => {agent}")

    # --- Si LLM detecta saludo, no hacer recomendación ---
    if intent == "greeting":
        return {
            "response": response_text,
            "recommendations": [],
            "detected_name": detected_name,
            "suggest_edit": False
        }

    # --- Si no es intent de recomendación, fallback ---
    if intent != "recommend_movies":
        # Si el LLM no genera una respuesta (porque el intent no es de recomendación),
        # usamos un texto por defecto para que el frontend no muestre una burbuja vacía.
        return {
            "response": response_text or "No he entendido bien tu petición. ¿Puedes reformularla?",
            "recommendations": [],
            "suggest_edit": True,
            "transcription": user_text
        }

    # --- Flujo de recomendación ---
    rec_result = await rec_service.get_enriched_recommendations(
        user_id="user_session",
        preferences=filters,
        user_query=filters.get("similar_to") or user_text,
        limit=filters.get("limit", 5)
    )
    
    recommendations = rec_result.get("results", [])
    match_type = rec_result.get("match_type", "exact")

    if not recommendations:
        return {
            "response": "Lo siento, no he encontrado ninguna película que coincida con tu búsqueda.",
            "recommendations": [],
            "suggest_edit": True,
            "transcription": user_text
        }

    # Si hubo coincidencia parcial, avisamos al usuario
    if match_type == "partial":
        response_text = "No encontré coincidencias exactas con todos tus filtros, pero aquí tienes algunas opciones relacionadas. " + response_text

    # Generar respuesta natural con los títulos usando el LLM
    # Esto permite que el asistente comente sobre las películas ("X es genial", "Y te va a encantar")
    if recommendations:
        # Usamos el nombre detectado ahora o el que ya teníamos en sesión
        name_to_use = detected_name if detected_name else (current_user_name if current_user_name != "Usuario" else None)
        response_text = generate_recommendation_response(user_text, recommendations, response_text, user_name=name_to_use)

    return {
        "response": response_text,
        "recommendations": recommendations,
        "suggest_edit": False
    }

@app.get("/health")
def health():
    return {"status": "ok"}
