"""
analysis_agent.py
==================

This module implements a predictive economic analysis agent for Argentina using
data from the Banco Central de la República Argentina (BCRA).  The agent is
capable of fetching time‑series data from the BCRA open API, forecasting
variables such as the ARS/USD exchange rate, evaluating credit risk for
individuals or companies via the Central de Deudores API and simulating
investment returns via Monte Carlo techniques.  It exposes a simple
command‑line interface so that analysts can experiment with different
modelling techniques without writing boilerplate code.

The core forecasting models implemented include:

* **ARIMA**: Autoregressive Integrated Moving Average models for univariate
  time series forecasting.  Implemented with `statsmodels`.
* **RandomForestRegressor**: An ensemble tree method from scikit‑learn, used
  here to capture nonlinear relationships in time series data.
* **Monte Carlo Simulation**: Generates random paths for asset returns
  following a normal distribution.  Useful for stress testing and portfolio
  optimisation.

Please note that an LSTM (Long Short‑Term Memory) model is included as a
stub below.  The current runtime environment does not ship with deep
learning frameworks such as TensorFlow or PyTorch, so the LSTM example
cannot be executed here.  However, the function demonstrates how one
could construct and train a recurrent neural network using Keras when
those dependencies are available.  Users with access to TensorFlow can
uncomment the code and use it to produce nonlinear forecasts.

When running this script as a stand‑alone program (see the `__main__`
section), it demonstrates the agent by generating a synthetic ARS/USD
exchange rate series, fitting the available models and producing forward
forecasts.  The Monte Carlo simulation uses the historical returns of the
synthetic series to illustrate how the agent can assess investment risk.

Disclaimer
----------
The forecasts and risk metrics produced by this agent are for educational
purposes only and do not constitute financial advice.  Users should
exercise caution and consult qualified professionals when making
investment decisions.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


class BCRAClient:
    """A simple wrapper around the BCRA time‑series API.

    The BCRA offers a catalogue of public APIs for accessing financial and
    economic statistics.  The time‑series API hosted at apis.datos.gob.ar
    exposes hundreds of series via a unified `/series` endpoint.  Clients may
    specify one or more series identifiers (`ids`) and optionally filter by
    dates, limit the number of records returned or request the last `n`
    observations.  See the API reference for full details.

    This client encapsulates common patterns such as building query strings
    and normalising the JSON response into a pandas DataFrame.
    """

    BASE_URL = "https://apis.datos.gob.ar/series/api/series/"

    def __init__(self):
        # Use a session so that HTTP connections are reused.
        self.session = requests.Session()

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        last: Optional[int] = None,
        limit: int = 1000,
        metadata: str = "none",
    ) -> pd.DataFrame:
        """Fetches a single time series from the BCRA API.

        Args:
            series_id: The identifier of the series to retrieve.  Use the
                `/search` endpoint on the API documentation to discover IDs.
            start_date: Optional ISO date (YYYY-MM-DD).  If provided, only
                observations on or after this date are returned.
            end_date: Optional ISO date (YYYY-MM-DD).  If provided, only
                observations up to this date are returned.
            last: If provided, returns the last N observations from the
                series.  Mutually exclusive with `limit` and `start_date`.
            limit: Maximum number of observations per request (max 1000).  If
                the series has more observations than `limit`, additional
                requests with `start` offsets are required; this helper will
                automatically page through all data.
            metadata: Level of metadata to return.  Defaults to "none" for
                efficiency.

        Returns:
            A DataFrame with columns `timestamp` and `value`, where
            `timestamp` is a pandas.Timestamp.

        Raises:
            requests.HTTPError: if the HTTP request fails.
        """
        if last is not None and (start_date or end_date):
            raise ValueError("`last` cannot be used with `start_date` or `end_date`.")

        # Build query parameters.
        params = {
            "ids": series_id,
            "format": "json",
            "metadata": metadata,
        }
        if last is not None:
            params["last"] = last
        else:
            params["limit"] = limit
            if start_date:
                params["start_date"] = start_date
            if end_date:
                params["end_date"] = end_date

        all_data: List[Tuple[str, float]] = []
        start = 0

        while True:
            if last is None:
                params["start"] = start
            response = self.session.get(self.BASE_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            data_points: List[List] = payload.get("data", [])
            all_data.extend([(dp[0], dp[1]) for dp in data_points])

            # Pagination: if fewer than `limit` observations were returned, we're done.
            if last is not None or len(data_points) < limit:
                break
            start += limit

        # Build DataFrame.
        df = pd.DataFrame(all_data, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


def generate_synthetic_exchange_rate(
    start: str = "2010-01-01",
    end: str = _dt.date.today().isoformat(),
    initial_rate: float = 3.0,
    drift: float = 0.0005,
    volatility: float = 0.01,
) -> pd.Series:
    """Generates a synthetic ARS/USD exchange rate series.

    Because network restrictions may prevent access to live BCRA data during
    development, this helper synthesises a plausible exchange rate series for
    demonstration.  It uses a geometric Brownian motion model with a small
    drift (depreciation) and daily volatility.  The resulting series is
    returned as a pandas Series indexed by date.

    Args:
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        initial_rate: The exchange rate at the start of the series.
        drift: Expected daily percentage change (e.g. 0.0005 ≈ 0.05% per day).
        volatility: Standard deviation of the daily percentage change.

    Returns:
        pd.Series of simulated exchange rates.
    """
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    # Generate daily returns via normal distribution.
    daily_returns = np.random.normal(loc=drift, scale=volatility, size=n)
    # Cumulative sum of returns plus initial rate.
    rate = initial_rate * np.exp(np.cumsum(daily_returns))
    return pd.Series(rate, index=dates, name="ars_usd")


def train_arima(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1), steps: int = 30) -> Tuple[np.ndarray, pd.Series]:
    """Fits an ARIMA model and produces forecasts.

    Args:
        series: Historical time series with a datetime index.
        order: ARIMA order (p, d, q).
        steps: Number of steps ahead to forecast.

    Returns:
        forecasts: Numpy array containing the forecasted values.
        conf_int: Pandas DataFrame with confidence intervals for each forecast.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast_res = model_fit.get_forecast(steps=steps)
    forecast_values = forecast_res.predicted_mean.values
    conf_int = forecast_res.conf_int(alpha=0.05)
    return forecast_values, conf_int


def train_random_forest(
    series: pd.Series,
    n_estimators: int = 200,
    lookback: int = 5,
    steps: int = 30,
) -> np.ndarray:
    """Trains a RandomForestRegressor on a univariate time series.

    The model uses a fixed lookback window of past values to predict the next
    observation.  For example, with `lookback=5`, each training sample
    consists of the previous five days' exchange rates and the target is the
    current day's rate.  After fitting, the model generates forecasts
    iteratively for the specified number of future steps.

    Args:
        series: Historical time series with a datetime index.
        n_estimators: Number of trees in the forest.
        lookback: Number of past observations used as features.
        steps: Number of steps ahead to forecast.

    Returns:
        Numpy array containing the forecasted values.
    """
    data = series.values
    X = []
    y = []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    # Optionally evaluate performance on the test set.
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest MSE on hold‑out set: {mse:.4f}")
    # Forecast iteratively.
    last_window = data[-lookback:].tolist()
    forecasts = []
    for _ in range(steps):
        next_pred = model.predict(np.array(last_window).reshape(1, -1))[0]
        forecasts.append(next_pred)
        last_window.pop(0)
        last_window.append(next_pred)
    return np.array(forecasts)


def monte_carlo_simulation(
    historical_returns: np.ndarray,
    num_simulations: int = 10000,
    horizon: int = 30,
    initial_value: float = 1.0,
) -> np.ndarray:
    """Runs a Monte Carlo simulation of asset returns.

    The simulation assumes that returns are independent and identically
    distributed.  Random draws are taken from the empirical distribution of
    historical returns to generate future price paths.  The resulting array
    contains the terminal value of the asset in each simulation.

    Args:
        historical_returns: Array of historical percentage returns (e.g.
            computed via `np.diff(series) / series[:-1]`).
        num_simulations: Number of random paths to simulate.
        horizon: Length of each simulated path (in time steps).
        initial_value: Starting value of the asset.

    Returns:
        Numpy array of shape (num_simulations,) with terminal asset values.
    """
    returns = historical_returns
    simulations = np.empty(num_simulations)
    for i in range(num_simulations):
        path_returns = np.random.choice(returns, size=horizon, replace=True)
        terminal_price = initial_value * np.prod(1 + path_returns)
        simulations[i] = terminal_price
    return simulations


def train_lstm(
    series: pd.Series,
    lookback: int = 10,
    epochs: int = 50,
    batch_size: int = 32,
    steps: int = 30,
) -> np.ndarray:
    """Trains an LSTM model for univariate time‑series forecasting.

    This function constructs a simple Long Short‑Term Memory (LSTM) network
    using the Keras API from TensorFlow.  It first normalises the input
    series and creates overlapping windows of length `lookback` to use as
    features.  The model learns to predict the next value in the sequence.
    After training, it generates `steps` forecasts iteratively.

    **Note:** TensorFlow/Keras is not installed in this environment.  The
    code below is provided for illustrative purposes and will raise an
    ImportError if executed here.  To run this function, install
    `tensorflow>=2.0` and `keras` in your Python environment.

    Args:
        series: Historical time series with a datetime index.
        lookback: Number of past observations to use as input features.
        epochs: Number of training epochs for the network.
        batch_size: Mini‑batch size for training.
        steps: Number of steps ahead to forecast.

    Returns:
        Numpy array containing the forecasted values.
    """
    try:
        import tensorflow as tf  # type: ignore[import]
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import LSTM as KerasLSTM, Dense  # type: ignore
        from sklearn.preprocessing import MinMaxScaler  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "LSTM functionality requires TensorFlow/Keras. "
            "Please install tensorflow>=2.0 to use this feature."
        ) from exc

    # Normalise the series to the range [0, 1].
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = series.values.reshape(-1, 1)
    scaled_values = scaler.fit_transform(values)
    # Prepare sequences.
    X = []
    y = []
    for i in range(lookback, len(scaled_values)):
        X.append(scaled_values[i - lookback : i, 0])
        y.append(scaled_values[i, 0])
    X = np.array(X)
    y = np.array(y)
    # Reshape input to [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build LSTM network.
    model = Sequential()
    model.add(KerasLSTM(units=50, return_sequences=False, input_shape=(lookback, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    # Train the model.
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    # Forecast iteratively.
    last_window = scaled_values[-lookback:].flatten().tolist()
    forecasts = []
    for _ in range(steps):
        window_input = np.array(last_window).reshape((1, lookback, 1))
        next_scaled = model.predict(window_input, verbose=0)[0, 0]
        # Append raw scale forecast to the list.
        forecasts.append(next_scaled)
        # Slide the window.
        last_window.pop(0)
        last_window.append(next_scaled)
    # Denormalise forecasts back to original scale.
    forecasts = np.array(forecasts).reshape(-1, 1)
    inv_forecasts = scaler.inverse_transform(forecasts)[:, 0]
    return inv_forecasts


def evaluate_credit_risk(identificacion: str) -> Dict[str, any]:
    """Fetches credit risk information for a given CUIT/CUIL/CDI.

    The Central de Deudores API returns the classification of a debtor along
    with the outstanding amount, days overdue and other flags.  This helper
    makes a call to the `centraldeudores` endpoint and summarises the risk.

    Note: Accessing this endpoint with a real identification number exposes
    sensitive personal information.  Use only with consent and ensure
    compliance with data protection laws.  For demonstration purposes,
    this function returns mock data when network calls are not permitted.

    Args:
        identificacion: A string representing the CUIT/CUIL/CDI (11 digits).

    Returns:
        A dictionary containing key risk metrics (situation, outstanding debt,
        days overdue, etc.).
    """
    try:
        url = f"https://api.bcra.gob.ar/CentralDeDeudores/v1.0/Deudas/{identificacion}"
        resp = requests.get(url)
        resp.raise_for_status()
        payload = resp.json().get("results", {})
        # Extract relevant fields.
        periods = payload.get("periodos", [])
        if periods:
            # Use the latest period.
            latest = periods[0]
            entities = latest.get("entidades", [])
            # Aggregate outstanding debt and find worst situation.
            total_debt = sum(e.get("monto", 0.0) for e in entities)
            worst_situation = max(e.get("situacion", 0) for e in entities) if entities else 0
            return {
                "identificacion": identificacion,
                "periodo": latest.get("periodo"),
                "total_debt": total_debt,
                "worst_situation": worst_situation,
            }
        else:
            return {
                "identificacion": identificacion,
                "periodo": None,
                "total_debt": 0.0,
                "worst_situation": 0,
            }
    except Exception:
        # Return mock data if the API is unreachable.
        random.seed(hash(identificacion) % (2**32))
        worst_situation = random.randint(1, 5)
        total_debt = round(random.uniform(0, 1000), 2)
        return {
            "identificacion": identificacion,
            "periodo": None,
            "total_debt": total_debt,
            "worst_situation": worst_situation,
            "note": "Mock data used due to network restrictions",
        }


def prepare_features_for_investment(series: pd.Series) -> np.ndarray:
    """Computes daily returns from a price series.

    Args:
        series: A pandas Series of prices.

    Returns:
        Numpy array of percentage changes.
    """
    returns = series.pct_change().dropna().values
    return returns


def demonstrate_agent() -> None:
    """Runs a demonstration of the predictive agent using synthetic data.

    This function synthesises an ARS/USD series, trains ARIMA and Random
    Forest models, runs a Monte Carlo simulation for investment risk and
    queries the credit risk evaluation function with a dummy CUIT.  Results
    are printed to the console.
    """
    print("Generating synthetic ARS/USD series...")
    synthetic_series = generate_synthetic_exchange_rate()
    # Use only the last 5 years for modelling.
    cutoff_date = synthetic_series.index[-1] - pd.DateOffset(years=5)
    recent_series = synthetic_series[synthetic_series.index >= cutoff_date]

    print("\nTraining ARIMA model...")
    arima_forecast, arima_conf = train_arima(recent_series, order=(2, 1, 2), steps=30)
    print("Next 5 ARIMA forecasts:", arima_forecast[:5])

    print("\nTraining Random Forest model...")
    rf_forecast = train_random_forest(recent_series, n_estimators=300, lookback=10, steps=30)
    print("Next 5 RF forecasts:", rf_forecast[:5])

    print("\nOptional: training LSTM model (may be skipped if TensorFlow is unavailable)...")
    try:
        lstm_forecast = train_lstm(recent_series, lookback=20, epochs=20, batch_size=16, steps=30)
        print("Next 5 LSTM forecasts:", lstm_forecast[:5])
    except ImportError as e:
        print("LSTM training skipped:", e)

    print("\nRunning Monte Carlo simulation...")
    returns = prepare_features_for_investment(recent_series)
    sims = monte_carlo_simulation(returns, num_simulations=5000, horizon=30)
    # Compute statistics.
    mean_terminal = np.mean(sims)
    var_5pct = np.percentile(sims, 5)
    print(f"Expected terminal value after 30 days: {mean_terminal:.4f}")
    print(f"5th percentile (VaR at 95%): {var_5pct:.4f}")

    print("\nEvaluating credit risk (mock)...")
    risk = evaluate_credit_risk("20123456789")
    print(risk)


if __name__ == "__main__":
    demonstrate_agent()