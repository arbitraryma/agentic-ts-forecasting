import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


def detect_seasonal_period(y, max_lag=60, min_period=2):
    y = pd.Series(y).dropna().astype(float)

    if len(y) < 2 * min_period:
        return None

    max_lag = min(max_lag, len(y) // 2)

    acf_values = acf(
        y,
        nlags=max_lag,
        fft=True
    )

    candidate_lags = []

    for lag in range(min_period, len(acf_values) - 1):
        if (
            acf_values[lag] > acf_values[lag - 1]
            and acf_values[lag] > acf_values[lag + 1]
        ):
            candidate_lags.append(lag)

    if not candidate_lags:
        return None

    best_lag = max(
        candidate_lags,
        key=lambda lag: acf_values[lag]
    )

    return int(best_lag)


def basic_diagnostics(y):
    y = pd.Series(y)

    missing_rate = y.isna().mean()

    y_clean = (
        y.dropna()
        .astype(float)
        .reset_index(drop=True)
    )

    n = len(y_clean)

    x = np.arange(n)

    slope = np.polyfit(
        x,
        y_clean,
        1
    )[0]

    seasonal_period = detect_seasonal_period(
        y_clean
    )

    return {
        "n_obs": n,
        "mean": float(y_clean.mean()),
        "std": float(y_clean.std()),
        "trend_slope": float(slope),
        "missing_rate": float(missing_rate),
        "seasonal_period": seasonal_period,
    }