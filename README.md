# voice-movie-recommendation-ai
Asistente conversacional multimodal para recomendación de cine. Integra Llama 3, Whisper y XTTS para interacción por voz en tiempo real. TFM Máster IA (UNIR).

# 🎬 Sistema de Recomendación de Cine por Voz con IA Generativa

> **Trabajo de Fin de Máster (TFM)** - Máster Universitario en Inteligencia Artificial (UNIR)  
> **Autor:** Juan Alberto Peña Núñez
> **Estado:** 🚀 Release Candidate (Febrero 2026)

---

## 📋 Descripción del Proyecto

Este proyecto representa la convergencia entre la ingeniería de software moderna y la Inteligencia Artificial Generativa. El objetivo principal ha sido desarrollar un **Asistente Conversacional Multimodal** que permita a los usuarios descubrir películas interactuando mediante voz natural, superando las limitaciones de los sistemas de recomendación tradicionales basados en filtros estáticos.

Como ingeniero Full-Stack con experiencia, el reto no ha sido únicamente la implementación de modelos de IA, sino la **arquitectura de una solución robusta, escalable y desacoplada**. El sistema orquesta múltiples modelos de *State-of-the-Art* (SOTA) ejecutándose localmente (*On-Premise*), garantizando privacidad y baja latencia sin depender de APIs de terceros para la inferencia.

## 🛠️ Stack Tecnológico & Arquitectura

La solución sigue un patrón de **microservicios contenerizados** orquestados con Docker, asegurando la reproducibilidad del entorno y el aislamiento de dependencias complejas (CUDA, Torch).

### 🧠 Core de IA (Local & Privado)
*   **Razonamiento & NLP:** `Llama 3.1 (8B)` ejecutándose sobre **Ollama**. Actúa como el "cerebro" agéntico que interpreta la intención del usuario y extrae entidades estructuradas (JSON) a partir de lenguaje natural.
*   **Speech-to-Text (STT):** `Faster-Whisper` (implementación optimizada CTranslate2) para transcripción de alta fidelidad y baja latencia.
*   **Text-to-Speech (TTS):** `Coqui XTTS v2`. Implementa clonación de voz *zero-shot*, permitiendo al asistente hablar con una voz clonada a partir de una referencia de pocos segundos.
*   **Búsqueda Semántica:** `Sentence-Transformers` para re-ranking de resultados basado en similitud vectorial de sinopsis.

### 🏗️ Ingeniería de Software
*   **Backend:** `FastAPI` (Python). Gestiona la lógica de negocio asíncrona, la orquestación de modelos y la integración con servicios externos.
*   **Frontend:** `Streamlit`. Interfaz de usuario reactiva con componentes personalizados (CSS/JS) para la captura de audio y visualización de medios.
*   **Datos:** Integración con API de **TMDB** (The Movie Database) + Caché de alto rendimiento en **Redis**.
*   **Infraestructura:** `Docker Compose` con soporte para `nvidia-container-toolkit` (GPU Passthrough).

## ✨ Características Clave

1.  **Interacción Multimodal Real**: Comunicación fluida por voz bidireccional. El sistema escucha, transcribe, piensa y responde con voz sintetizada.
2.  **Clonación de Voz Personalizada**: Módulo que permite clonar voces en tiempo real. Sube un audio de referencia y el asistente adoptará esa identidad sonora.
3.  **Búsqueda Híbrida Inteligente**: Combina filtrado determinista (Año, Género, Actor) con búsqueda semántica vectorial para entender peticiones abstractas (ej: *"Películas que se sientan como un sueño febril"*).
4.  **Gestión Dinámica de Recursos**: Selector en tiempo real para conmutar la inferencia entre CPU y GPU, permitiendo evaluar el impacto en latencia y consumo de VRAM.
5.  **Resiliencia y UX**: Manejo de errores robusto, edición manual de transcripciones y feedback visual de métricas del sistema.

## 🚀 Instalación y Despliegue

El proyecto está diseñado para ser agnóstico del entorno, aunque se recomienda encarecidamente una GPU NVIDIA (VRAM >= 8GB) para una experiencia fluida.

### Prerrequisitos
*   Docker & Docker Compose
*   NVIDIA Container Toolkit (para aceleración GPU)
*   API Key de TMDB

### Pasos
1.  **Clonar el repositorio:**
    ```bash
    git clone <repo-url>
    cd movie_recommendation_voice
    ```

2.  **Configuración:**
    Crea un archivo `.env` en la raíz con tus credenciales:
    ```env
    TMDB_API_KEY=tu_api_key_aqui
    ```

3.  **Despliegue:**
    ```bash
    docker-compose up -d --build
    ```
    *Nota: La primera ejecución descargará los pesos de los modelos (Llama 3, Whisper, XTTS), lo cual puede tomar varios minutos.*

4.  **Acceso:**
    *   Frontend: `http://localhost:8501`
    *   Documentación API (Swagger): `http://localhost:8000/docs`

## 📂 Estructura del Repositorio

```text
.
├── backend/                 # Microservicio Principal (FastAPI)
│   ├── models/              # Wrappers para Whisper, TTS, NLP
│   ├── services/            # Lógica de negocio (Agente LLM, TMDB, Recomendador)
│   └── main.py              # Entrypoint y Endpoints
├── frontend/                # Interfaz de Usuario (Streamlit)
│   └── app.py               # Lógica de UI y gestión de estado
├── nlp_service/             # Microservicio auxiliar (Soporte NLP)
├── docker-compose.yml       # Orquestación de contenedores
└── README.md                # Documentación del proyecto
```

## 🎓 Conclusión

Este TFM demuestra la viabilidad técnica de construir asistentes de voz avanzados y personalizados utilizando exclusivamente herramientas Open Source. La arquitectura propuesta equilibra la complejidad de los modelos de IA generativa con las mejores prácticas de ingeniería de software, resultando en un producto mantenible, escalable y funcional.

---
*Proyecto desarrollado para la Universidad Internacional de La Rioja (UNIR).*
