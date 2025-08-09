import os
import time
import threading
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException, Request, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

# ======================
# Config
# ======================
API_KEY = os.getenv("API_KEY")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
BCRA_BASE = os.getenv("BCRA_BASE_URL", "https://api.bcra.gob.ar")

_rate_buckets: Dict[str, list] = {}
_jobs: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Economic Agent API (Hybrid-Pro)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# Helpers
# ======================
def _require_api_key(x_api_key: Optional[str]):
    # Si no definiste API_KEY en el entorno, no exige clave (modo dev)
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _rate_limit(key: str):
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = time.time()
    bucket = _rate_buckets.setdefault(key, [])
    # limpiar ventana de 60s
    while bucket and now - bucket[0] > 60:
        bucket.pop(0)
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)

# análisis
try:
    import analysis_agent
except Exception:
    analysis_agent = None

class RunConfig(BaseModel):
    horizon: int = 30
    models: list[str] = ["arima", "random_forest", "monte_carlo"]
    series: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    async_mode: bool = False

# ======================
# Health & Metrics
# ======================
@app.get("/health")
def health():
    return {"status": "ok"}

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

# ======================
# BCRA Passthroughs
# ======================
def _bcra_get(path: str, params: Optional[Dict[str, Any]] = None):
    url = BCRA_BASE.rstrip("/") + "/" + path.lstrip("/")
    r = requests.get(url, params=params, timeout=30)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Propagar detalle del BCRA
        raise HTTPException(status_code=r.status_code, detail=r.text or str(e))
    try:
        return r.json()
    except Exception:
        return {"status": "ok", "body": r.text}

@app.get("/bcra/principales-variables")
def principales_variables_list(
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    # Ejemplo: catálogo de variables monetarias (ruta de ejemplo de BCRA)
    return _bcra_get("/estadisticas/v3.0/Monetarias")

@app.get("/bcra/principales-variables/{variable_id}")
def principales_variables_data(
    variable_id: int,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get(f"/estadisticas/v3.0/Monetarias/{variable_id}/series")

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
    cuit: str = Query(...),
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get(f"/deudores/v1.0/informe/{cuit}")

@app.get("/bcra/passthrough")
def bcra_passthrough(
    path: str = Query(...),
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    return _bcra_get(path)

# ======================
# Agent Runner
# ======================
def _run_agent_sync(cfg: RunConfig) -> Dict[str, Any]:
    if analysis_agent is None:
        raise RuntimeError("analysis_agent not available")

    if hasattr(analysis_agent, "demonstrate_agent"):
        try:
            return analysis_agent.demonstrate_agent(
                horizon=cfg.horizon,
                models=cfg.models,
                series=cfg.series,
                notes=cfg.notes
            )
        except TypeError:
            # Compatibilidad con firmas antiguas sin kwargs
            return analysis_agent.demonstrate_agent()

    raise RuntimeError("demonstrate_agent not found in analysis_agent")

def _worker_thread(job_id: str, cfg: RunConfig):
    _jobs[job_id]["status"] = "running"
    try:
        res = _run_agent_sync(cfg)
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = res
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)

@app.post("/run")
def run_agent(
    cfg: RunConfig,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")

    if cfg.async_mode:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {"status": "queued", "config": cfg.dict()}
        t = threading.Thread(target=_worker_thread, args=(job_id, cfg), daemon=True)
        t.start()
        return {"ok": True, "jobId": job_id, "status": "queued"}
    else:
        res = _run_agent_sync(cfg)
        return {"ok": True, "result": res}

@app.get("/status/{job_id}")
def job_status(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None),
    request: Request = None
):
    _require_api_key(x_api_key); _rate_limit(request.client.host if request and request.client else "anon")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job
