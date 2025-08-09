import os
import time
import threading
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Request, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import requests
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

app = FastAPI(title="Economic Agent API", version="1.0")

# CORS habilitado para pruebas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY", "economicagent")

# Almacén de trabajos asíncronos en memoria
jobs: Dict[str, Dict[str, Any]] = {}

class RunConfig(BaseModel):
    horizon: int = 30
    models: list[str] = ["arima", "random_forest"]
    series: Optional[list[float]] = None
    notes: Optional[str] = None
    async_mode: bool = False

# -------------------------
# Funciones internas
# -------------------------

def generate_synthetic_series(n=100):
    rng = np.random.default_rng()
    series = np.cumsum(rng.normal(loc=0.01, scale=0.02, size=n)) + 18
    return series

def train_arima(series, horizon):
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast.tolist()

def train_random_forest(series, horizon):
    X = np.arange(len(series)).reshape(-1, 1)
    y = np.array(series)
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X, y)
    future_X = np.arange(len(series), len(series) + horizon).reshape(-1, 1)
    forecast = model.predict(future_X)
    return forecast.tolist()

def monte_carlo_simulation(series, horizon, n_simulations=1000):
    last_value = series[-1]
    rng = np.random.default_rng()
    returns = np.diff(series) / series[:-1]
    simulated_terminal_values = []
    for _ in range(n_simulations):
        simulated_returns = rng.choice(returns, size=horizon, replace=True)
        simulated_price = last_value * np.prod(1 + simulated_returns)
        simulated_terminal_values.append(simulated_price)
    expected = np.mean(simulated_terminal_values) / last_value
    var_95 = np.percentile(simulated_terminal_values, 5) / last_value
    return expected, var_95

def demonstrate_agent(config: RunConfig):
    # Generar serie (real o simulada)
    if config.series:
        series = config.series
    else:
        series = generate_synthetic_series()

    # Ejecutar modelos solicitados
    arima_preds = train_arima(series, config.horizon) if "arima" in config.models else None
    rf_preds = train_random_forest(series, config.horizon) if "random_forest" in config.models else None

    # Simulación Monte Carlo
    mc_expected, mc_var = monte_carlo_simulation(series, config.horizon)

    # DEVOLVER CAMPOS FINALES
    return {
        "arima_forecasts": [float(x) for x in arima_preds] if arima_preds else None,
        "random_forest_forecasts": [float(x) for x in rf_preds] if rf_preds else None,
        "montecarlo": {
            "expected_terminal_value": float(mc_expected),
            "var_95": float(mc_var)
        },
        "timestamp": datetime.now().isoformat()
    }

# -------------------------
# Endpoints
# -------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(config: RunConfig, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    if config.async_mode:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued", "config": config.dict(), "result": None}

        def background_job():
            jobs[job_id]["status"] = "running"
            result = demonstrate_agent(config)
            jobs[job_id]["result"] = result
            jobs[job_id]["status"] = "done"

        threading.Thread(target=background_job).start()
        return {"jobId": job_id}
    else:
        result = demonstrate_agent(config)
        return {"ok": True, "result": result}

@app.get("/status/{job_id}")
def status(job_id: str, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

