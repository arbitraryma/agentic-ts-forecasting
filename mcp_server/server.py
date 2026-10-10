"""
FastMCP server.

Every tool calls services.forecasting_service or the
dataset loader. This module does not import FastAPI
and does not reimplement forecasting math.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from fastmcp import FastMCP

from benchmarks.datasets import load_dataset as load_registered_dataset
from services.forecasting_service import (
    diagnose,
    forecast_from_series,
    get_candidate_models as get_candidate_models_service,
    make_forecast,
    select_expert as select_expert_service,
    validate,
)


mcp = FastMCP("agentic-ts-forecasting")


def _jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(
        value,
        bool,
    ):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    return str(value)


@mcp.tool
def load_dataset(
    dataset_name: str,
    data_root: str = "data",
) -> dict:
    """Load a registered dataset and return its numeric target."""

    dataset = load_registered_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
    )
    return _jsonable({
        "dataset_name": dataset["name"],
        "n_obs": int(len(dataset["y"])),
        "values": dataset["y"].astype(float).tolist(),
        "frequency": dataset["frequency"],
        "known_seasonal_period": dataset[
            "known_seasonal_period"
        ],
        "n_series": dataset["n_series"],
        "has_covariates": dataset["has_covariates"],
        "metadata": dataset["metadata"],
    })


@mcp.tool
def diagnose_series(
    values: list[float],
    seasonal_period: int | None = None,
) -> dict:
    """Run deterministic diagnostics on one numeric series."""

    return _jsonable(
        diagnose(
            values,
            seasonal_period=seasonal_period,
        )
    )


@mcp.tool
def get_candidate_models(
    values: list[float],
    horizon: int,
    compute_budget: str = "low",
    n_series: int = 1,
    has_covariates: bool = False,
    seasonal_period: int | None = None,
) -> dict:
    """Return the candidate models allowed for this series."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    return _jsonable(
        get_candidate_models_service(
            values,
            horizon=horizon,
            compute_budget=compute_budget,
            n_series=n_series,
            has_covariates=has_covariates,
            seasonal_period=seasonal_period,
        )
    )


@mcp.tool
def run_validation(
    values: list[float],
    horizon: int,
    candidates: list[str],
    n_windows: int = 1,
    fast: bool = True,
) -> dict:
    """Run rolling-origin validation for the given candidates."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not candidates:
        raise ValueError("candidates must not be empty.")

    return _jsonable(
        validate(
            values,
            horizon=horizon,
            candidates=candidates,
            n_windows=n_windows,
            fast=fast,
            verbose=False,
        )
    )


@mcp.tool
def select_expert(
    validation_summary: list[dict],
    compute_weight: float = 0.10,
    temperature: float = 1.0,
) -> dict:
    """Select the hard expert from validation rows already computed."""

    if not validation_summary:
        raise ValueError(
            "validation_summary must not be empty."
        )

    return _jsonable(
        select_expert_service(
            validation_summary,
            compute_weight=compute_weight,
            temperature=temperature,
        )
    )


@mcp.tool
def generate_forecast(
    values: list[float],
    model_name: str,
    window_type: str,
    horizon: int,
    seasonal_period: int | None = None,
) -> dict:
    """Forecast with one already selected model and window."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    diagnostics = diagnose(
        values,
        seasonal_period=seasonal_period,
    )
    forecast = make_forecast(
        values,
        diagnostics,
        model_name,
        window_type,
        horizon,
        fast=True,
        n_series=1,
    )
    return _jsonable({
        "model": model_name,
        "window_type": window_type,
        "horizon": int(horizon),
        "forecast": forecast,
    })


@mcp.tool
def run_full_forecasting_pipeline(
    values: list[float],
    horizon: int,
    target_name: str = "series",
    compute_budget: str = "low",
    models: list[str] | None = None,
    seasonal_period: int | None = None,
) -> dict:
    """Run diagnostics, validation, routing, and forecasting."""

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    return _jsonable(
        forecast_from_series(
            values,
            horizon=horizon,
            target_name=target_name,
            seasonal_period=seasonal_period,
            compute_budget=compute_budget,
            model_allowlist=models,
            fast=True,
            n_windows=1,
            verbose=False,
        )
    )


def main():
    mcp.run()


if __name__ == "__main__":
    main()
