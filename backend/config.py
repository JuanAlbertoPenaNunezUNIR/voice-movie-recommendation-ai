# Configuración centralizada para el microservicio backend

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del servicio backend."""
    
    # Servicio
    SERVICE_NAME: str = "movie-recommendation-backend"
    VERSION: str = "2.0.0"
    
    # Base de datos
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./recommendation.db")
    
    # Servicios externos
    NLP_SERVICE_URL: str = os.getenv("NLP_SERVICE_URL", "http://nlp_service:8001")
    TMDB_API_KEY: Optional[str] = os.getenv("TMDB_API_KEY")
    
    # Modelos
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "distil-large-v3")
    WHISPER_DEVICE: Optional[str] = os.getenv("WHISPER_DEVICE")  # auto, cpu, cuda
    WHISPER_COMPUTE_TYPE: Optional[str] = os.getenv("WHISPER_COMPUTE_TYPE")  # int8, float16
    
    # Cache
    TRANSFORMERS_CACHE: str = os.getenv("TRANSFORMERS_CACHE", "/app/.cache/transformers")
    HF_HOME: str = os.getenv("HF_HOME", "/app/.cache/huggingface")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/ai_system.log")
    
    # CORS
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Redis (opcional)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", "redis://redis:6379")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Instancia global de configuración
settings = Settings()

