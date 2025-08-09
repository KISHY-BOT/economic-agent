# Hybrid-Pro: API + Worker (Railway) para Agente Económico

Este paquete agrega una **API FastAPI securizada** y mantiene un **worker** 24/7.
- Seguridad: **API Key** (`X-API-Key`), rate limit por IP, CORS configurable.
- Operación: sync o async (`/run` con `async_mode:true`) + `/status/{jobId}`.
- Observabilidad: `/metrics` (texto estilo Prometheus).
- BCRA: wrappers a endpoints públicos (sin auth BCRA).

## Archivos
- `app.py` → API (FastAPI).
- `worker.py` → bucle de análisis periódico.
- `Procfile` → define `web` y `worker`.
- `openapi.gpt.yaml` → schema para Actions de GPT (con API Key).
- `requirements.additions.txt` → dependencias a añadir a tu `requirements.txt`.

## Deploy en Railway
1) Conectar repo → New Project → Deploy from GitHub.
2) Variables de entorno:
   - `API_KEY=<clave_segura>` (recomendado)
   - `RATE_LIMIT_PER_MIN=60`
   - `BCRA_BASE_URL=https://api.bcra.gob.ar` (opcional)
   - `RUN_ONCE=false`, `RUNTIME_INTERVAL_SECONDS=21600` (worker en loop)
   - `CORS_ALLOW_ORIGINS=*` (o lista separada por comas)
3) Procfile:
```
web: uvicorn app:app --host 0.0.0.0 --port ${PORT}
worker: python worker.py
```

## Pruebas rápidas
- Salud: `GET /health`
- Ejecutar sync:
```
curl -X POST https://YOUR_RAILWAY_URL/run   -H "Content-Type: application/json"   -H "X-API-Key: YOUR_KEY"   -d '{"horizon":60,"models":["arima","random_forest"],"async_mode":false}'
```
- Ejecutar async:
```
curl -X POST https://YOUR_RAILWAY_URL/run   -H "Content-Type: application/json"   -H "X-API-Key: YOUR_KEY"   -d '{"horizon":60,"async_mode":true}'
# luego
curl https://YOUR_RAILWAY_URL/status/JOB_ID -H "X-API-Key: YOUR_KEY"
```
- BCRA:
```
curl https://YOUR_RAILWAY_URL/bcra/estadisticas-cambiarias/cotizaciones -H "X-API-Key: YOUR_KEY"
```
