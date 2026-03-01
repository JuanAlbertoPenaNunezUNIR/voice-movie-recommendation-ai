# Contenedor docker para el servicio de procesamiento de lenguaje natural (NLP) con GPU
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Variables de entorno para optimizar Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias de sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Establecer python3.10 como el por defecto
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero para cachear dependencias
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip

# Instalar PyTorch con soporte CUDA y componentes adicionales
RUN python3 -m pip install torch==2.5.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir \
      "faster-whisper>=1.0.0" \
      "av>=12.0.0,<17.0.0" \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación (excluye archivos de .dockerignore)
COPY . .

# Exponer puerto del servicio
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]