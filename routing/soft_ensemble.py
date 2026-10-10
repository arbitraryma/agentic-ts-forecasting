import numpy as np
import pandas as pd


def normalize_weights(weights):
    """
    Normalize non-negative weights so that
    they sum to 1.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if np.any(weights < 0):
        raise ValueError(
            "Ensemble weights must be non-negative."
        )

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "Ensemble weights must sum to a "
            "positive value."
        )

    return weights / total


def weighted_forecast(
    forecasts,
    weights,
):
    """
    Compute a weighted ensemble forecast.

    Parameters
    ----------
    forecasts : array-like
        Shape:

        (n_models, horizon)

    weights : array-like
        Shape:

        (n_models,)

    Returns
    -------
    np.ndarray
        Shape:

        (horizon,)
    """

    forecasts = np.asarray(
        forecasts,
        dtype=float,
    )

    weights = normalize_weights(
        weights
    )

    if forecasts.ndim != 2:
        raise ValueError(
            "forecasts must have shape "
            "(n_models, horizon)."
        )

    if forecasts.shape[0] != len(
        weights
    ):
        raise ValueError(
            "Number of forecasts does not "
            "match number of weights."
        )

    ensemble = np.sum(
        forecasts
        * weights[:, None],
        axis=0,
    )

    return ensemble


def build_ensemble_table(
    expert_names,
    weights,
):
    """
    Build a readable table of ensemble weights.
    """

    weights = normalize_weights(
        weights
    )

    return pd.DataFrame({
        "expert": expert_names,
        "ensemble_weight": weights,
    }).sort_values(
        "ensemble_weight",
        ascending=False,
    ).reset_index(
        drop=True
    )