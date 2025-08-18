from __future__ import annotations

import os
import time
import ssl
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ===== Config =====
BCRA_BASE = os.getenv("BCRA_BASE_URL", "https://api.bcra.gob.ar")
BCRA_TIMEOUT = int(os.getenv("BCRA_TIMEOUT", "60"))
BCRA_VERIFY_SSL = os.getenv("BCRA_VERIFY_SSL", "true").lower() != "false"
CA_BUNDLE = os.getenv("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

# Configurar logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuración SSL
ssl_context = ssl.create_default_context()
ssl_context.load_verify_locations(cafile=CA_BUNDLE)
logger.info(f"TLS(analysis_agent) OpenSSL: {ssl.OPENSSL_VERSION}")
logger.info(f"TLS(analysis_agent) VERIFY_ENABLED={BCRA_VERIFY_SSL}, VERIFY_PARAM={CA_BUNDLE}")

# ====== Utils BCRA v3 ======
def _bcra_get_monetaria(
id_var: int,
desde: Optional[str] = None,
hasta: Optional[str] = None,
timeout: int = BCRA_TIMEOUT,
) -> List[Dict[str, Any]]:
"""
Trae una serie monetaria del BCRA v3.0 con manejo robusto de SSL
Retorna [{'fecha': 'YYYY-MM-DD', 'valor': number}, ...]
"""
url = f"{BCRA_BASE}/estadisticas/v3.0/monetarias/{id_var}"
params: Dict[str, Any] = {}
if desde: params["desde"] = desde
if hasta: params["hasta"] = hasta

# Configuración SSL
verify_param = CA_BUNDLE if BCRA_VERIFY_SSL else False

# Reintentos para fallas SSL/red
for attempt in range(3):
try:
r = requests.get(
url,
params=params,
timeout=timeout,
verify=verify_param,
stream=True
)
r.raise_for_status()
data = r.json()
return data if isinstance(data, list) else []
except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
if attempt == 2:
logger.error(f"BCRA SSL/Connection error after 3 attempts: {str(e)}")
raise
logger.warning(f"BCRA SSL/Connection error (attempt {attempt+1}): {str(e)}")
time.sleep(1) # Esperar antes de reintentar
except requests.HTTPError as e:
logger.error(f"BCRA HTTP error: {e.response.status_code if e.response else 'No response'}")
raise

def _latest_value(series: List[Dict[str, Any]]) -> Optional[float]:
return series[-1]["valor"] if series else None

def _to_pd_series(series: List[Dict[str, Any]], name: str = "valor") -> pd.Series:
if not series:
return pd.Series(dtype=float)
df = pd.DataFrame(series)
# Normaliza nombres por si la API cambia mayúsculas:
fecha_col = "fecha" if "fecha" in df.columns else "Fecha"
valor_col = "valor" if "valor" in df.columns else "Valor"
df[fecha_col] = pd.to_datetime(df[fecha_col])
df = df.sort_values(fecha_col)
s = pd.Series(df[valor_col].values, index=df[fecha_col].values, name=name)
s.index = pd.to_datetime(s.index)
s = s.asfreq("D") # serie diaria (rellena NaN en días sin dato)
s = s.ffill() # forward-fill
return s

# ====== Modelos ======
def _forecast_arima_pd(series: pd.Series, horizon: int) -> List[float]:
"""
ARIMA simple con statsmodels si está disponible.
Si no está instalado, retorna lista vacía.
"""
try:
from statsmodels.tsa.arima.model import ARIMA # type: ignore
except Exception as e:
logger.warning(f"ARIMA no disponible: {str(e)}")
return []
if series.dropna().empty:
return []
# Orden conservador; podés tunearlo
model = ARIMA(series.dropna(), order=(1, 1, 1))
fit = model.fit()
fc = fit.get_forecast(steps=horizon).predicted_mean
return [float(x) for x in fc.values.tolist()]

def _forecast_rf_pd(series: pd.Series, horizon: int, lookback: int = 30) -> List[float]:
"""
Random Forest univariante (features = últimas 'lookback' observaciones).
Si sklearn no está disponible, retorna lista vacía.
"""
try:
from sklearn.ensemble import RandomForestRegressor # type: ignore
except Exception as e:
logger.warning(f"scikit-learn no disponible: {str(e)}")
return []
y = series.dropna().values
if len(y) < lookback + 1:
logger.warning(f"Datos insuficientes para Random Forest: {len(y)} < {lookback+1}")
return []

# Ajustar lookback si hay pocos datos
lookback = min(lookback, len(y) // 2)

X, t = [], []
for i in range(lookback, len(y)):
X.append(y[i - lookback : i])
t.append(y[i])
X = np.array(X)
t = np.array(t)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, t)

# Forecast iterativo
window = y[-lookback:].tolist()
out: List[float] = []
for _ in range(horizon):
pred = float(model.predict(np.array(window).reshape(1, -1))[0])
out.append(pred)
window.pop(0)
window.append(pred)
return out

def _forecast_montecarlo_pd(series: pd.Series, horizon: int, num_sims: int = 5000) -> Dict[str, Any]:
"""
Monte Carlo simple con bootstrap de retornos diarios.
Devuelve métricas básicas (esperado y VaR 95).
"""
s = series.dropna()
if len(s) < 10:
logger.warning("Datos insuficientes para Monte Carlo")
return {"expected_terminal_value": None, "var_95": None}

rets = s.pct_change().dropna().values
if len(rets) < 5:
logger.warning("Retornos insuficientes para Monte Carlo")
return {"expected_terminal_value": None, "var_95": None}

last = float(s.iloc[-1])
terminals = []
for _ in range(num_sims):
path = np.random.choice(rets, size=horizon, replace=True)
terminals.append(last * np.prod(1 + path))
arr = np.array(terminals)
return {
"expected_terminal_value": float(np.mean(arr)),
"var_95": float(np.percentile(arr, 5)),
}

# ====== Entrada del agente (lo que llama app.py) ======
def get_inputs() -> Dict[str, Any]:
"""
Obtiene insumos mínimos para el modelo.
Oficial (minorista promedio) id=7.
MEP: no disponible directo en BCRA → None.
"""
oficial_series = _bcra_get_monetaria(7)
oficial_last = _latest_value(oficial_series)
ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
return {
"oficial_series": oficial_series,
"oficial_last": oficial_last,
"mep_last": None, # Derivación MEP por bonos no implementada aquí
"timestamp": ts,
}

def demonstrate_agent(
horizon: int = 30,
models: Optional[List[str]] = None,
series: Optional[Dict[str, Any]] = None,
notes: Optional[str] = None,
) -> Dict[str, Any]:
"""
Punto de entrada usado por app.py.
- Usa datos reales BCRA v3 para el dólar oficial (id 7).
- Retorna un dict JSON-serializable con inputs, outputs y notas.
- Si faltan librerías de modelos, no rompe: deja listas vacías.
"""
inputs = get_inputs()
if not inputs["oficial_last"]:
raise ValueError("No se pudo obtener el último valor del dólar oficial (id 7).")

selected = models or ["arima", "random_forest", "monte_carlo"]
# Permite inyectar series desde app si se quisiera (prioridad a 'series')
oficial_series = series.get("oficial_series") if series and "oficial_series" in series else inputs["oficial_series"]
s = _to_pd_series(oficial_series, name="usd_oficial")

result: Dict[str, Any] = {
"inputs": {
"oficial_last": inputs["oficial_last"],
"mep_last": inputs["mep_last"],
"timestamp": inputs["timestamp"],
},
"params": {"horizon": horizon, "models": selected, "notes": notes},
"timestamp": inputs["timestamp"],
}

warnings: List[str] = []

if "arima" in selected:
ar = _forecast_arima_pd(s, horizon)
if not ar:
warnings.append("ARIMA no ejecutado (falta statsmodels o datos insuficientes).")
result["arima_forecasts"] = ar

if "random_forest" in selected:
rf = _forecast_rf_pd(s, horizon)
if not rf:
warnings.append("Random Forest no ejecutado (falta scikit-learn o datos insuficientes).")
result["random_forest_forecasts"] = rf

if "monte_carlo" in selected:
mc = _forecast_montecarlo_pd(s, horizon)
result["montecarlo"] = mc

# Nota por MEP ausente
result["notice"] = "MEP no disponible vía BCRA; se proyecta solo el oficial. Si aportás series de bonos (AL30/GD30) puedo derivarlo."
if warnings:
result["warnings"] = warnings

return result
