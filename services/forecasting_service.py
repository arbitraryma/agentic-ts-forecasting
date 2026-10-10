"""
Shared forecasting service.

FastAPI, FastMCP, and the LangGraph forecasting
node call these functions. The numerical work is
delegated to agent.tools, which wraps the existing
diagnostics, validation, and routing modules.
"""

import io

import pandas as pd

from agent.tools import (
    diagnose_series,
    generate_forecast,
    get_candidate_models as get_candidate_models_tool,
    run_full_forecasting_pipeline,
    run_validation,
    select_expert as select_expert_tool,
)


def _series(values):
    numeric = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    ).dropna()

    if len(numeric) < 8:
        raise ValueError(
            "The series is empty or too short after "
            "numeric coercion."
        )

    return numeric.astype(float).reset_index(drop=True)


def _apply_known_period(diagnostics, seasonal_period):
    """Same known-period rule as the benchmark runner."""

    if (
        seasonal_period is not None
        and diagnostics.get("seasonal_period") is None
    ):
        diagnostics = dict(diagnostics)
        diagnostics["seasonal_period"] = int(
            seasonal_period
        )

    return diagnostics


def diagnose(values, seasonal_period=None):
    """Diagnostics for one numeric series."""

    diagnostics = diagnose_series(_series(values))
    return _apply_known_period(
        diagnostics,
        seasonal_period,
    )


def get_candidate_models(
    values,
    horizon,
    compute_budget="low",
    n_series=1,
    has_covariates=False,
    seasonal_period=None,
):
    """Candidate models for one series. Routing stays in Python."""

    diagnostics = diagnose(
        values,
        seasonal_period=seasonal_period,
    )
    routed = get_candidate_models_tool(
        diagnostics=diagnostics,
        horizon=int(horizon),
        n_series=n_series,
        has_covariates=has_covariates,
        compute_budget=compute_budget,
    )
    return {
        "diagnostics": diagnostics,
        "candidates": routed["candidates"],
        "reasons": routed["reasons"],
    }


def validate(
    values,
    horizon,
    candidates,
    diagnostics=None,
    **settings,
):
    """Inner rolling validation for an explicit candidate list."""

    series = _series(values)
    if diagnostics is None:
        diagnostics = diagnose(series)

    return run_validation(
        y=series,
        diagnostics=diagnostics,
        candidates=list(candidates),
        horizon=int(horizon),
        n_windows=settings.get("n_windows", 1),
        window_types=settings.get("window_types"),
        train_window=settings.get("train_window", 60),
        fast=settings.get("fast", True),
        n_series=settings.get("n_series", 1),
        verbose=settings.get("verbose", False),
    )


def select_expert(
    validation_summary,
    compute_weight=0.10,
    temperature=1.0,
):
    """Dynamic hard expert selection."""

    return select_expert_tool(
        validation_summary=validation_summary,
        compute_weight=compute_weight,
        temperature=temperature,
    )


def make_forecast(
    values,
    diagnostics,
    model_name,
    window_type,
    horizon,
    **settings,
):
    """Refit one already selected expert and forecast."""

    return generate_forecast(
        y=_series(values),
        diagnostics=diagnostics,
        model_name=model_name,
        window_type=window_type,
        horizon=int(horizon),
        fast=settings.get("fast", True),
        n_series=settings.get("n_series", 1),
        train_window=settings.get("train_window", 60),
    )


def forecast_from_series(
    values,
    horizon,
    target_name="series",
    seasonal_period=None,
    **settings,
):
    """
    Full deterministic pipeline for an in-memory series.

    target_name is recorded for the caller. It is not
    used as a model feature.
    """

    series = _series(values)
    horizon = int(horizon)

    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if horizon >= len(series):
        raise ValueError(
            "horizon must be shorter than the series."
        )

    result = run_full_forecasting_pipeline(
        y=series,
        horizon=horizon,
        n_series=settings.get("n_series", 1),
        has_covariates=settings.get(
            "has_covariates",
            False,
        ),
        compute_budget=settings.get(
            "compute_budget",
            "low",
        ),
        n_windows=settings.get("n_windows", 1),
        window_types=settings.get("window_types"),
        train_window=settings.get("train_window", 60),
        fast=settings.get("fast", True),
        compute_weight=settings.get(
            "compute_weight",
            0.10,
        ),
        temperature=settings.get("temperature", 1.0),
        known_seasonal_period=seasonal_period
        or settings.get("known_seasonal_period"),
        model_allowlist=settings.get("model_allowlist"),
        verbose=settings.get("verbose", False),
    )
    result["target_name"] = target_name
    return result


def series_from_csv_bytes(content, filename, target_column):
    """Read one numeric column from an uploaded CSV."""

    if not str(filename).lower().endswith(".csv"):
        raise ValueError("Upload must be a .csv file.")

    if not content:
        raise ValueError("CSV file is empty.")

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "CSV file must be UTF-8 text."
        ) from exc

    frame = pd.read_csv(io.StringIO(text))

    if target_column not in frame.columns:
        columns = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Target column '{target_column}' was not "
            f"found. Columns: {columns}"
        )

    return _series(frame[target_column].tolist())


def forecast_from_csv(
    content,
    filename,
    target_column,
    horizon,
    **settings,
):
    series = series_from_csv_bytes(
        content,
        filename,
        target_column,
    )
    return forecast_from_series(
        series.tolist(),
        horizon=horizon,
        target_name=target_column,
        **settings,
    )


def forecast_from_dataset(
    dataset_name,
    horizon,
    data_root="data",
    **settings,
):
    """Forecast a registered dataset. Paths come from the loader."""

    from benchmarks.datasets import load_dataset

    dataset = load_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        target=settings.get("target"),
    )
    return forecast_from_series(
        dataset["y"].tolist(),
        horizon=horizon,
        target_name=dataset["metadata"].get(
            "target",
            dataset_name,
        ),
        seasonal_period=dataset.get(
            "known_seasonal_period"
        ),
        n_series=dataset["n_series"],
        has_covariates=dataset["has_covariates"],
        **{
            key: value
            for key, value in settings.items()
            if key != "target"
        },
    )
