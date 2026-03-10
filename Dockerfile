# Usamos una imagen base oficial de Python ligera
FROM python:3.11-slim

# Actualizar pip
RUN pip install --upgrade pip

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos primero el archivo de dependencias (para optimizar caché de Docker)
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código fuente del proyecto a /app
COPY . .

# Exponemos el puerto en el que correrá FastAPI (por defecto 8000)
EXPOSE 9000

# Comando para ejecutar la aplicación con Gunicorn (multi-worker)
# Es CRUCIAL usar host "0.0.0.0" para que los port-forward de Docker y Coolify funcionen hacia el exterior.
CMD ["gunicorn", "main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:9000"]
