# Backend para el sistema de recomendación de películas por voz

# Base CUDA si hay GPU; si no, funciona en CPU igualmente
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Variables de entorno
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Aceptar términos Coqui TTS
ENV COQUI_TOS_AGREED=1  

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3.11-venv \
    python3-pip \
    build-essential \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Eliminar Python 3.10 por conflictos y forzar Python 3.11 por defecto
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Instalar pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# PyTorch + Torchaudio compatibles con TTS y CUDA 12.1
RUN python3.11 -m pip install \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Instalar transformers compatible ANTES
RUN python3.11 -m pip install transformers==4.35.2

# Instalar Coqui SIN resolver deps
RUN python3.11 -m pip install TTS==0.22.0 --no-deps

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements primero para cache
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Descargar modelo SBERT en build time
RUN python3.11 - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
EOF

# Copiar código de la app
COPY . .

# Exponer puerto FastAPI
EXPOSE 8000

# CMD para iniciar Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 