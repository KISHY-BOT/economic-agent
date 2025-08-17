from __future__ import annotations

import os
import logging
import ssl
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import certifi
import numpy as np
import pandas as pd
import requests

# Modelos
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------
# Logging básico
# ---------------------------------------------------------------------
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------
BCRA_BASE = os.getenv("BCRA_BASE_URL", "https://api.bcra.gob.ar")
BCRA_TIMEOUT = int(os.getenv("BCRA_TIMEOUT", "60"))
BCRA_VERIFY_SSL = os.getenv("BCRA_VERIFY_SSL", "true").lower() != "false"

SYSTEM_BUNDLE = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE")
VERIFY_PARAM = (SYSTEM_BUNDLE or certifi.where()) if BCRA_VERIFY_SSL else False


def _log_tls_env(prefix: str = "TLS(analysis_agent)"):
    """Logs de diagnóstico TLS (no rompe si algo falta)."""
    try:
        log.info("%s OpenSSL: %s", prefix, getattr(ssl, "OPENSSL_VERSION", "<unknown>"))
    except Exception as e:
        log.warning("%s OpenSSL: <error> %s", prefix, e)
    try:
        import requests.certs as rc
        log.info("%s certifi.where(): %s", prefix, certifi.where())
        log.info("%s requests.certs.where(): %s", prefix, rc.where())
    except Exception as e:
        log.warning("%s cert paths error: %s", prefix, e)
    try:
        dvp = ssl.get_default_verify_paths()
        log.info("%s default verify paths: cafile=%s capath=%s", prefix, dvp.cafile, dvp.capath)
    except Exception as e:
        log.warning("%s ssl.get_default_verify_paths() error: %s", prefix, e)
    log.info("%s VERIFY_ENABLED=%s, VERIFY_PARAM=%s", prefix, BCRA_VERIFY_SSL, VERIFY_PARAM)


# ---------------------------------------------------------------------
# Fetch BCRA
# ---------------------------------------------------------------------
def _bcra_get_monetaria(
    id_var: int,
    desde: Optional[str] = None,
    hasta: Optional[str] = None,
    timeout: int = BCRA_TIMEOUT,
) -> List[Dict[str, Any]]:
    """
    /estadisticas/v3.0/monetarias/{id}
    Retorna [{'fecha': 'YYYY-MM-DD', 'valor': number}, ...]
    """
    url = f"{BCRA_BASE.rstrip('/')}/estadisticas/v3.0/monetarias/{id_var}"
    params: Dict[str, Any] = {}
    if desde:
        params["desde"] = desde
    if hasta:
        params["hasta"] = hasta

    r = requests.get(url, params=params, timeout=timeout, verify=VERIFY_PARAM)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------
# Transformaciones
# ---------------------------------------------------------------------
def _to_pd_series(series: List[Dict[str, Any]], name: str = "valor") -> pd.Series:
    """
    Convierte [{'fecha','valor'}...] a pd.Series diaria, forward-filled.
    """
    if not series:
        return pd.Series(dtype=float)

    df = pd.DataFrame(series)
    # Normalizar columnas por si la API cambia el casing
    fecha_col = "fecha" if "fecha" in df.columns else ("Fecha" if "Fecha" in df.columns else None)
    valor_col = "valor" if "valor" in df.columns else ("Valor" if "Valor" in df.columns else None)
    if not fecha_col or not valor_col:
        raise ValueError("Estructura inesperada en datos del BCRA")

    df[fecha_col] = pd.to_datetime(df[fecha_col])
    df = df.sort_values(fecha_col)
    s = pd.Series(df[valor_col].values, index=df[fecha_col].values, name=name)
    s.index = pd.to_datetime(s.index)
    # Frecuencia diaria y forward-fill
    s = s.asfreq("D").ffill()
    # Convertir explícitamente a float (evita object dtype si vienen strings)
    s = s.astype(float)
    return s


def _as_dict_series(s: pd.Series) -> List[Dict[str, Any]]:
    """Convierte Serie a [{'fecha': 'YYYY-MM-DD', 'valor': float}, ...]"""
    out: List[Dict[str, Any]] = []
    for idx, v in s.items():
        # idx puede ser Timestamp/DatetimeIndex
        out.append({"fecha": str(pd.to_datetime(idx).date()), "valor": float(v)})
    return out


# ---------------------------------------------------------------------
# Modelos de pronóstico
# ---------------------------------------------------------------------
def _forecast_arima_pd(s: pd.Series, horizon: int = 30, order: Tuple[int, int, int] = (1, 1, 1)) -> pd.Series:
    """
    ARIMA simple (p,d,q). Ajusta sobre la serie completa y proyecta 'horizon' pasos diarios.
    """
    if len(s.dropna()) < 10:
        raise ValueError("Serie insuficiente para ARIMA (min ~10 observaciones).")
    model = ARIMA(s.astype(float), order=order)
    fitted = model.fit()
    pred = fitted.forecast(steps=horizon)  # devuelve Serie con índice enteros
    # Reindexar con fechas consecutivas
    last_date = pd.to_datetime(s.index[-1])
    future_idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
    pred.index = future_idx
    pred.name = "arima_forecast"
    return pred


def _features_from_series(s: pd.Series, lags: int = 7) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construye features de lags para un regresor (RandomForest).
    X: [lag1..lagN], y: valor actual.
    """
    df = pd.DataFrame({"y": s.astype(float)})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["y"].shift(i)
    df = df.dropna()
    y = df["y"].copy()
    X = df.drop(columns=["y"]).copy()
    return X, y


def _forecast_rf_pd(s: pd.Series, horizon: int = 30, lags: int = 14, n_estimators: int = 400) -> pd.Series:
    """
    Random Forest autoregresivo con lags. Pronóstico recursivo h pasos.
    """
    if len(s.dropna()) < (lags + 10):
        raise ValueError(f"Serie insuficiente para RF (min ~{lags+10} observaciones).")

    s = s.astype(float).copy()
    X, y = _features_from_series(s, lags=lags)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=123,
        n_jobs=-1
    )
    rf.fit(X, y)

    # Predicción recursiva
    hist = s.copy()
    preds: List[float] = []
    last_date = pd.to_datetime(hist.index[-1])

    for step in range(horizon):
        # construir vector de entrada con últimos lags
        arr = []
        for i in range(1, lags + 1):
            arr.append(float(hist.iloc[-i]))
        X_new = np.array(arr, dtype=float).reshape(1, -1)
        y_hat = float(rf.predict(X_new)[0])
        preds.append(y_hat)
        # extender la serie histórica con el pronóstico
        hist.loc[last_date + timedelta(days=step + 1)] = y_hat

    idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
    pred = pd.Series(preds, index=idx, name="rf_forecast")
    return pred


def _forecast_montecarlo_pd(
    s: pd.Series,
    horizon: int = 30,
    n_sims: int = 1000,
    use_log_returns: bool = True,
    clamp_nonnegative: bool = True,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Simulación Monte Carlo con retornos empíricos (bootstrap).
    - Si use_log_returns: simula log-retornos y aplica compounding exponencial.
    - Retorna la mediana por día y bandas P10/P90.
    """
    if len(s.dropna()) < 30:
        raise ValueError("Serie insuficiente para Monte Carlo (min ~30 observaciones).")

    s = s.astype(float)
    last_val = float(s.iloc[-1])

    # calcular retornos
    if use_log_returns:
        rets = np.log(s / s.shift(1)).dropna().values
    else:
        rets = s.pct_change().dropna().values

    if rets.size == 0:
        raise ValueError("No se pudieron calcular retornos para Monte Carlo.")

    # simulación
    sims = np.zeros((n_sims, horizon), dtype=float)
    rng = np.random.default_rng(123)
    for i in range(n_sims):
        path_rets = rng.choice(rets, size=horizon, replace=True)
        if use_log_returns:
            # log-compounding
            path_vals = last_val * np.exp(np.cumsum(path_rets))
        else:
            # simple compounding
            path_vals = last_val * np.cumprod(1.0 + path_rets)
        if clamp_nonnegative:
            path_vals = np.maximum(path_vals, 0.0)
        sims[i, :] = path_vals

    # estadísticas por día
    p50 = np.median(sims, axis=0)
    p10 = np.percentile(sims, 10, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    last_date = pd.to_datetime(s.index[-1])
    idx = pd.date_range(last_date + timedelta(days=1), periods=horizon, freq="D")
    median_series = pd.Series(p50, index=idx, name="mc_median")

    bands = {
        "p10_first": float(p10[0]),
        "p90_first": float(p90[0]),
        "p10_last": float(p10[-1]),
        "p90_last": float(p90[-1]),
    }
    return median_series, bands


# ---------------------------------------------------------------------
# Orquestador demo (llamado por worker/app)
# ---------------------------------------------------------------------
def demonstrate_agent(
    horizon: int = 30,
    models: Optional[List[str]] = None,
    series: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orquesta la obtención de datos y pronósticos.
    - models: lista entre ['arima','random_forest','monte_carlo'] (orden no importa).
    - series: si None, trae BCRA id=7 como demo. Si viene:
        { "id_var": 7, "desde": "YYYY-MM-DD", "hasta": "YYYY-MM-DD" }
      o bien { "data": [{"fecha":"YYYY-MM-DD","valor":...}, ...] } para usar datos custom.
    """
    _log_tls_env()

    models = models or ["arima", "random_forest", "monte_carlo"]

    # 1) Obtener/armar la serie base
    if series and isinstance(series, dict) and "data" in series:
        raw = series["data"]
    else:
        id_var = 7 if not series else int(series.get("id_var", 7))
        desde = series.get("desde") if series else None
        hasta = series.get("hasta") if series else None
        raw = _bcra_get_monetaria(id_var=id_var, desde=desde, hasta=hasta)

    s = _to_pd_series(raw, name="serie_base")

    result: Dict[str, Any] = {
        "meta": {
            "source": "BCRA" if "data" not in (series or {}) else "custom",
            "horizon_days": horizon,
            "models": models,
            "last_value": float(s.iloc[-1]) if len(s) else None,
            "last_date": str(pd.to_datetime(s.index[-1]).date()) if len(s) else None,
            "notes": notes,
        },
        "input_len": int(len(s)),
        "forecasts": {},
    }

    if len(s) == 0:
        result["error"] = "Serie vacía"
        return result

    # 2) Ejecutar modelos solicitados
    for m in models:
        m = m.lower().strip()
        try:
            if m == "arima":
                arima_fc = _forecast_arima_pd(s, horizon=horizon)
                result["forecasts"]["arima"] = _as_dict_series(arima_fc)

            elif m in ("random_forest", "rf"):
                rf_fc = _forecast_rf_pd(s, horizon=horizon)
                result["forecasts"]["random_forest"] = _as_dict_series(rf_fc)

            elif m in ("monte_carlo", "mc"):
                mc_med, bands = _forecast_montecarlo_pd(s, horizon=horizon, n_sims=1000)
                result["forecasts"]["monte_carlo"] = {
                    "median": _as_dict_series(mc_med),
                    "bands": bands,
                }

            else:
                result["forecasts"][m] = {"error": "modelo no reconocido"}

        except Exception as e:
            log.exception("Error en modelo %s: %s", m, e)
            result["forecasts"][m] = {"error": str(e)}

    return result


# Ejecuta diagnóstico TLS al importar (no es crítico si falla)
try:
    _log_tls_env()
except Exception:
    pass
