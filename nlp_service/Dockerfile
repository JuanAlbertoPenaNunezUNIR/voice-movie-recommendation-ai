# Contenedor docker para el servicio NLP con GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Variables de entorno Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ----------------------------
# Dependencias del sistema
# ----------------------------
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    ffmpeg \
    git \
    curl \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Hacer python3.10 default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Directorio de trabajo
WORKDIR /app

# ----------------------------
# Pip actualizado
# ----------------------------
RUN python -m pip install --upgrade pip

# ----------------------------
# Instalar PyTorch CUDA 12.1
# ----------------------------
RUN python -m pip install --no-cache-dir torch==2.2.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

# ----------------------------
# Copiar requirements Y instalar dependencias base
# ----------------------------
COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

# ----------------------------
# Instalar Coqui TTS sin deps externas
# (evita conflictos con numpy, torch ya instalados)
# ----------------------------
RUN python -m pip install --no-cache-dir git+https://github.com/coqui-ai/TTS.git

# Instalar PyAV para audio/video processing (necesario para TTS)
RUN python -m pip install --no-cache-dir 'av>=12.0.0,<17.0.0'

# ----------------------------
# Copiar código de la app
# ----------------------------
COPY . .

# Exponer puerto
EXPOSE 8001

# Lanzar servicio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]