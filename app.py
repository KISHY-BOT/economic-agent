import os
import time
import threading
import uuid
import json
import atexit
import logging
import re
import certifi
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Request, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# ======================
# Helpers mejorados
# ======================
def _normalize_base(url: str) -> str:
    u = (url or "").strip()
    if not re.match(r"^https?://", u, flags=re.I):
        # si falta el esquema, forzamos https://
        u = "https://" + u.lstrip("/")
    return u

def _require_api_key(x_api_key: Optional[str]):
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _rate_limit(key: str):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    bucket = _rate_buckets.setdefault(key, [])
    while bucket and now - bucket[0] > 60:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)
    
# ======================
# Configuración inicial
# ======================
BCRA_VERIFY_SSL = os.getenv("BCRA_VERIFY_SSL", "true").lower() != "false"
logger.info(f"BCRA verify SSL: {BCRA_VERIFY_SSL}")

# Configurar logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Validar variables críticas
API_KEY = os.getenv("API_KEY")
BCRA_BASE = os.getenv("BCRA_BASE_URL", "https://api.bcra.gob.ar")
BCRA_TIMEOUT = int(os.getenv("BCRA_TIMEOUT", "60"))

if not BCRA_BASE:
    logger.error("BCRA_BASE_URL no está configurada")
    raise RuntimeError("BCRA_BASE_URL no configurada")

# Normalizar y loguear la base del BCRA
_BCRA_BASE = _normalize_base(BCRA_BASE)
logger.info(f"BCRA_BASE_URL (env): {BCRA_BASE!r}")
logger.info(f"BCRA_BASE efectivo usado: {_BCRA_BASE!r}")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

# Persistencia de jobs
JOBS_FILE = "jobs.json"

_rate_buckets: Dict[str, list] = {}
_jobs: Dict[str, Dict[str, Any]] = {}

# Cargar jobs persistentes
try:
    if os.path.exists(JOBS_FILE):
        with open(JOBS_FILE, "r") as f:
            _jobs = json.load(f)
        logger.info(f"Cargados {len(_jobs)} jobs desde disco")
except Exception as e:
    logger.error(f"Error cargando jobs: {str(e)}")

# Guardar jobs al salir
def _save_jobs():
    try:
        with open(JOBS_FILE, "w") as f:
            json.dump(_jobs, f)
        logger.info(f"Jobs guardados: {len(_jobs)}")
    except Exception as e:
        logger.error(f"Error guardando jobs: {str(e)}")

atexit.register(_save_jobs)

app = FastAPI(title="Economic Agent API (Hybrid-Pro)", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Análisis
try:
    import analysis_agent
except Exception as e:
    logger.error(f"Error importing analysis_agent: {str(e)}")
    analysis_agent = None

class RunConfig(BaseModel):
    horizon: int = 30
    models: list[str] = ["arima", "random_forest", "monte_carlo"]
    series: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    async_mode: bool = False

# ======================
# BCRA Passthroughs Corregidos
# ======================
def _bcra_get(path: str, params: Optional[Dict[str, Any]] = None):
    url = _BCRA_BASE.rstrip("/") + "/" + path.lstrip("/")
    try:
        logger.info(f"Request to BCRA: {url}")
        verify_param = certifi.where() if BCRA_VERIFY_SSL else False
        r = requests.get(url, params=params, timeout=BCRA_TIMEOUT, verify=verify_param)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return r.json()
        else:
            return {"content": r.text, "content_type": content_type}

    except requests.exceptions.SSLError as e:
        logger.error(f"BCRA SSL error: {e}")
        raise HTTPException(
            status_code=502,
            detail="Fallo de verificación SSL con BCRA. "
                   "Revise el trust store o defina BCRA_VERIFY_SSL=false para diagnóstico temporal."
        )

    except requests.HTTPError as e:
        error_detail = r.text if r.text else str(e)
        if len(error_detail) > 500:
            error_detail = error_detail[:500] + "... [truncated]"

        logger.error(f"BCRA error {r.status_code}: {error_detail}")

        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Recurso no encontrado")
        else:
            raise HTTPException(
                status_code=502,
                detail=f"Error BCRA [{r.status_code}]: {error_detail}"
            )

    except requests.Timeout:
        logger.error("BCRA timeout")
        raise HTTPException(status_code=504, detail="Timeout al conectar con BCRA")

    except Exception as e:
        logger.error(f"BCRA connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

# Rutas BCRA actualizadas según documentación oficial
@app.get("/bcra/monetarias/{id_var}")
def monetarias(
    id_var: int,
    desde: Optional[str] = None,
    hasta: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    params = {}
    if desde: params["desde"] = desde
    if hasta: params["hasta"] = hasta
    return _bcra_get(f"/estadisticas/v3.0/monetarias/{id_var}", params=params or None)

@app.get("/bcra/principales-variables")
def principales_variables_list(
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get("/estadisticas/v1.0/principalesvariables")

@app.get("/bcra/principales-variables/{variable_id}")
def principales_variables_data(
    variable_id: int,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get(f"/estadisticas/v1.0/principalesvariables/{variable_id}")

@app.get("/bcra/estadisticas-cambiarias/cotizaciones")
def cambiarias_cotizaciones(
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get("/estadisticas/v1.0/Cotizaciones")

@app.get("/bcra/cheques-denunciados")
def cheques_denunciados(
    numero: Optional[str] = None,
    cuit: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    params = {}
    if numero: params["numero"] = numero
    if cuit: params["cuit"] = cuit
    return _bcra_get("/cheques/v1.0/cheques", params=params or None)

@app.get("/bcra/deudores")
def deudores(
    cuit: str = Query(..., regex=r"^\d{2}-\d{8}-\d{1}$"),
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    # Limpiar formato CUIT (20-12345678-9 → 20123456789)
    clean_cuit = cuit.replace("-", "")
    return _bcra_get(f"/deudores/v1.0/informe/{clean_cuit}")

@app.get("/bcra/passthrough")
def bcra_passthrough(
    path: str = Query(...),
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get(path)

# ======================
# Health & Metrics
# ======================
@app.get("/health")
def health():
    return {"status": "ok", "bcra_status": "active"}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    total_jobs = len(_jobs)
    running = sum(1 for j in _jobs.values() if j.get("status") == "running")
    done = sum(1 for j in _jobs.values() if j.get("status") == "done")
    failed = sum(1 for j in _jobs.values() if j.get("status") == "failed")

    lines = [
        "# HELP econ_jobs_total Total jobs seen",
        "# TYPE econ_jobs_total gauge",
        f"econ_jobs_total {total_jobs}",
        "# HELP econ_jobs_running Jobs currently running",
        "# TYPE econ_jobs_running gauge",
        f"econ_jobs_running {running}",
        "# HELP econ_jobs_done Jobs finished successfully",
        "# TYPE econ_jobs_done gauge",
        f"econ_jobs_done {done}",
        "# HELP econ_jobs_failed Jobs failed",
        "# TYPE econ_jobs_failed gauge",
        f"econ_jobs_failed {failed}",
    ]
    return "\n".join(lines)

@app.get("/metrics.json")
def metrics_json():
    total_jobs = len(_jobs)
    running = sum(1 for j in _jobs.values() if j.get("status") == "running")
    done = sum(1 for j in _jobs.values() if j.get("status") == "done")
    failed = sum(1 for j in _jobs.values() if j.get("status") == "failed")
    return {
        "econ_jobs_total": total_jobs,
        "econ_jobs_running": running,
        "econ_jobs_done": done,
        "econ_jobs_failed": failed
    }

# ======================
# Agent Runner Mejorado
# ======================
def _run_agent_sync(cfg: RunConfig) -> Dict[str, Any]:
    if analysis_agent is None:
        raise RuntimeError("analysis_agent no disponible")

    if hasattr(analysis_agent, "demonstrate_agent"):
        try:
            return analysis_agent.demonstrate_agent(
                horizon=cfg.horizon,
                models=cfg.models,
                series=cfg.series,
                notes=cfg.notes
            )
        except TypeError:
            return analysis_agent.demonstrate_agent()

    raise RuntimeError("Función demonstrate_agent no encontrada")

def _worker_thread(job_id: str, cfg: RunConfig):
    try:
        _jobs[job_id]["status"] = "running"
        res = _run_agent_sync(cfg)
        _jobs[job_id] = {
            "status": "done",
            "result": res,
            "completed_at": time.time()
        }
    except Exception as e:
        _jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": time.time()
        }
    finally:
        _save_jobs()

@app.post("/run")
def run_agent(
    cfg: RunConfig,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")

    if cfg.async_mode:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "status": "queued",
            "config": cfg.dict(),
            "created_at": time.time()
        }
        threading.Thread(
            target=_worker_thread,
            args=(job_id, cfg),
            daemon=True
        ).start()
        _save_jobs()
        return {"job_id": job_id, "status": "queued"}
    else:
        return {"result": _run_agent_sync(cfg)}

@app.get("/status/{job_id}")
def job_status(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job no encontrado")
    return _jobs[job_id]
    
