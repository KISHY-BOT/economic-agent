import os
import time
import threading
import uuid
import json
import atexit
import logging
import re
import ssl
import certifi
from typing import Optional, Dict, Any
from datetime import datetime

import requests
from fastapi import FastAPI, Header, HTTPException, Request, Query, Path
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def _normalize_base(url: str) -> str:
    u = (url or "").strip()
    if not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u.lstrip("/")
    return u

def _validate_iso_date(date_str: str) -> bool:
    """Valida que una cadena tenga el formato YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("API_KEY")
BCRA_BASE = os.getenv("BCRA_BASE_URL", "https://api.bcra.gob.ar")
BCRA_TIMEOUT = int(os.getenv("BCRA_TIMEOUT", "60"))
BCRA_VERIFY_SSL = os.getenv("BCRA_VERIFY_SSL", "true").lower() != "false"

logger.info(f"BCRA verify SSL: {BCRA_VERIFY_SSL}")
if not BCRA_BASE:
    logger.error("BCRA_BASE_URL no está configurada")
    raise RuntimeError("BCRA_BASE_URL no configurada")

_BCRA_BASE = _normalize_base(BCRA_BASE)
logger.info(f"BCRA_BASE_URL (env): {BCRA_BASE!r}")
logger.info(f"BCRA_BASE efectivo usado: {_BCRA_BASE!r}")

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

JOBS_FILE = "jobs.json"
_rate_buckets: Dict[str, list] = {}
_jobs: Dict[str, Dict[str, Any]] = {}

try:
    if os.path.exists(JOBS_FILE):
        with open(JOBS_FILE, "r") as f:
            _jobs = json.load(f)
        logger.info(f"Cargados {len(_jobs)} jobs desde disco")
except Exception as e:
    logger.error(f"Error cargando jobs: {str(e)}")

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

def _log_tls_env(prefix: str = "TLS(app)"):
    try:
        logger.info("%s OpenSSL: %s", prefix, getattr(ssl, "OPENSSL_VERSION", "<unknown>"))
    except Exception as e:
        logger.warning("%s OpenSSL: <error> %s", prefix, e)
    try:
        import requests.certs as rc
        logger.info("%s certifi.where(): %s", prefix, certifi.where())
        logger.info("%s requests.certs.where(): %s", prefix, rc.where())
    except Exception as e:
        logger.warning("%s cert paths error: %s", prefix, e)
    try:
        dvp = ssl.get_default_verify_paths()
        logger.info("%s default verify paths: cafile=%s capath=%s", prefix, dvp.cafile, dvp.capath)
    except Exception as e:
        logger.warning("%s ssl.get_default_verify_paths() error: %s", prefix, e)
    system_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
    verify_param = (system_bundle or certifi.where()) if BCRA_VERIFY_SSL else False
    logger.info("%s VERIFY_ENABLED=%s, VERIFY_PARAM=%s", prefix, BCRA_VERIFY_SSL, verify_param)

@app.on_event("startup")
def on_startup():
    _log_tls_env()

def _bcra_get(path: str, params: Optional[Dict[str, Any]] = None):
    url = _BCRA_BASE.rstrip("/") + "/" + path.lstrip("/")
    try:
        logger.info(f"Request to BCRA: {url}")
        system_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
        verify_param = (system_bundle or certifi.where()) if BCRA_VERIFY_SSL else False

        r = requests.get(url, params=params, timeout=BCRA_TIMEOUT, verify=verify_param)
        r.raise_for_status()

        content_type = r.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return r.json()
        return r.text

    except requests.exceptions.SSLError as e:
        logger.error(f"BCRA SSL error: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"BCRA HTTP error: {e}")
        raise

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    total = len(_jobs)
    running = sum(1 for j in _jobs.values() if j.get("status") == "running")
    done = sum(1 for j in _jobs.values() if j.get("status") == "done")
    failed = sum(1 for j in _jobs.values() if j.get("status") == "failed")
    lines = [
        f"econ_jobs_total {total}",
        f"econ_jobs_running {running}",
        f"econ_jobs_done {done}",
        f"econ_jobs_failed {failed}",
    ]
    return "\n".join(lines) + "\n"

@app.get("/metrics.json")
def metrics_json():
    total = len(_jobs)
    running = sum(1 for j in _jobs.values() if j.get("status") == "running")
    done = sum(1 for j in _jobs.values() if j.get("status") == "done")
    failed = sum(1 for j in _jobs.values() if j.get("status") == "failed")
    return {
        "econ_jobs_total": total,
        "econ_jobs_running": running,
        "econ_jobs_done": done,
        "econ_jobs_failed": failed,
    }

# ======================
# BCRA endpoints existentes
# ======================
@app.get("/bcra/principales-variables")
def principales_variables():
    try:
        return _bcra_get("estadisticas/v1.0/principalesvariables")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/principales-variables/{id_var}")
def principales_variable(id_var: int = Path(...)):
    try:
        return _bcra_get(f"estadisticas/v1.0/principalesvariables/{id_var}")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/cheques/entidades")
def cheques_entidades():
    try:
        return _bcra_get("cheques/v1.0/entidades")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

# ======================
# BCRA stubs adicionales (ajustados a docs oficiales)
# ======================

@app.get("/bcra/cheques/denunciados/{codigo_entidad}/{numero_cheque}")
def cheques_denunciados(
    codigo_entidad: int = Path(..., ge=0),
    numero_cheque: int = Path(..., ge=0),
):
    """
    Cheques denunciados (según doc BCRA):
    GET /cheques/v1.0/denunciados/{codigoEntidad}/{numeroCheque}
    """
    try:
        return _bcra_get(f"cheques/v1.0/denunciados/{codigo_entidad}/{numero_cheque}")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/deudores/{identificacion}")
def deudores_por_identificacion(identificacion: str = Path(..., pattern=r"^\d{11}$")):
    """
    Central de Deudores (según doc BCRA):
    GET /CentralDeDeudores/v1.0/Deudas/{identificacion}  (11 dígitos)
    """
    try:
        return _bcra_get(f"CentralDeDeudores/v1.0/Deudas/{identificacion}")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/estadisticas-cambiarias/maestros/divisas")
def cambiarias_maestros_divisas():
    """
    Maestro de monedas (según doc BCRA):
    GET /estadisticascambiarias/v1.0/Maestros/Divisas
    """
    try:
        return _bcra_get("estadisticascambiarias/v1.0/Maestros/Divisas")
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/estadisticas-cambiarias/cotizaciones")
def cambiarias_cotizaciones(
    fecha: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    """
    Cotizaciones por fecha (según doc BCRA):
    GET /estadisticascambiarias/v1.0/Cotizaciones?fecha=yyyy-MM-dd
    """
    try:
        params = {}
        if fecha:
            if not _validate_iso_date(fecha):
                raise HTTPException(status_code=400, detail="fecha inválida (YYYY-MM-DD)")
            params["fecha"] = fecha
        return _bcra_get("estadisticascambiarias/v1.0/Cotizaciones", params=params or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/estadisticas-cambiarias/cotizaciones/{moneda}")
def cambiarias_evolucion(
    moneda: str = Path(..., min_length=3, max_length=3),
    fechadesde: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    fechahasta: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    limit: Optional[int] = Query(default=None, ge=10, le=1000),
    offset: Optional[int] = Query(default=None, ge=0),
):
    """
    Evolución de moneda (según doc BCRA):
    GET /estadisticascambiarias/v1.0/Cotizaciones/{moneda}?fechadesde=&fechahasta=&limit=&offset=
    """
    try:
        params = {}
        if fechadesde:
            if not _validate_iso_date(fechadesde):
                raise HTTPException(status_code=400, detail="fechadesde inválida (YYYY-MM-DD)")
            params["fechadesde"] = fechadesde
        if fechahasta:
            if not _validate_iso_date(fechahasta):
                raise HTTPException(status_code=400, detail="fechahasta inválida (YYYY-MM-DD)")
            params["fechahasta"] = fechahasta
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return _bcra_get(f"estadisticascambiarias/v1.0/Cotizaciones/{moneda.upper()}", params=params or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/monetarias/{id_var}")
def monetarias_v3(
    id_var: int = Path(..., ge=0),
    desde: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    hasta: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    limit: Optional[int] = Query(default=None, ge=1, le=3000),
    offset: Optional[int] = Query(default=None, ge=0),
):
    """
    Serie monetaria v3.0 por ID con filtro de fechas/paginado (según doc BCRA):
    GET /estadisticas/v3.0/monetarias/{IdVariable}?desde=&hasta=&limit=&offset=
    """
    try:
        params = {}
        if desde:
            if not _validate_iso_date(desde):
                raise HTTPException(status_code=400, detail="desde inválida (YYYY-MM-DD)")
            params["desde"] = desde
        if hasta:
            if not _validate_iso_date(hasta):
                raise HTTPException(status_code=400, detail="hasta inválida (YYYY-MM-DD)")
            params["hasta"] = hasta
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return _bcra_get(f"estadisticas/v3.0/monetarias/{id_var}", params=params or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

@app.get("/bcra/passthrough")
def bcra_passthrough(
    path: str = Query(..., description="Path comenzando con '/', ej: /estadisticas/v3.0/monetarias/7")
):
    """
    Proxy genérico a cualquier path público del BCRA.
    """
    p = (path or "").strip()
    if not p.startswith("/"):
        raise HTTPException(status_code=400, detail="path debe comenzar con '/'")
    try:
        return _bcra_get(p.lstrip("/"))
    except Exception as e:
        raise HTTPException(status_code=502, detail="Upstream BCRA error")

# ======================
# Agent runner (sync/async) — sin cambios
# ======================
class AgentResult(BaseModel):
    model_config = {"extra": "allow"}

def _run_agent_sync(cfg: RunConfig):
    if analysis_agent and hasattr(analysis_agent, "demonstrate_agent"):
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
        threading.Thread(target=_worker_thread, args=(job_id, cfg), daemon=True).start()
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
