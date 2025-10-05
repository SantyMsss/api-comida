# Usa Python 3.10 basado en Debian Bullseye (estable)
FROM python:3.10-slim-bullseye

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements primero (aprovechar cache de Docker)
COPY requirements.txt ./

# Forzar uso de HTTPS en los repositorios de Debian y luego instalar dependencias
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list \
    && sed -i 's|http://security.debian.org|https://security.debian.org|g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
       libgl1-mesa-glx \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       libgomp1 \
       libgtk-3-0 \
       libavcodec-dev \
       libavformat-dev \
       libswscale-dev \
       libv4l-dev \
       libxvidcore-dev \
       libx264-dev \
       libjpeg-dev \
       libpng-dev \
       libtiff-dev \
       libatlas-base-dev \
       python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer puerto 80
EXPOSE 9000

# Definir variable de entorno opcional
ENV NAME=World

# Comando de inicio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]