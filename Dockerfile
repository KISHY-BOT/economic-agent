# Dockerfile definitivo para Economic Agent
FROM python:3.11.9-slim

# 1. Actualizar certificados CA e instalar dependencias
RUN apt-get update && \
apt-get install -y --no-install-recommends \
ca-certificates \
curl \
openssl && \
update-ca-certificates --fresh && \
rm -rf /var/lib/apt/lists/* && \
# Descargar certificados raíz de Let's Encrypt
curl -sSf https://letsencrypt.org/certs/isrgrootx1.pem -o /usr/local/share/ca-certificates/isrgrootx1.crt && \
curl -sSf https://letsencrypt.org/certs/lets-encrypt-r3.pem -o /usr/local/share/ca-certificates/lets-encrypt-r3.crt && \
update-ca-certificates

# 2. Configurar entorno SSL y zona horaria
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV TZ=America/Argentina/Buenos_Aires
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 3. Configuración de Python
ENV PYTHONUNBUFFERED=1 \
PYTHONDONTWRITEBYTECODE=1 \
PIP_NO_CACHE_DIR=on

# 4. Directorio de trabajo y dependencias
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar aplicación
COPY . .

# 6. Healthcheck y puerto
HEALTHCHECK --interval=30s --timeout=3s \
CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000

# 7. Comando de inicio
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
