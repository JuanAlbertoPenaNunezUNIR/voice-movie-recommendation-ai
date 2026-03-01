# Contenedor docker para la interfaz de usuario con Streamlit
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar herramientas necesarias para compilar dependencias científicas si las ruedas no están disponibles
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando de ejecución con opciones optimizadas
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]