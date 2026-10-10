import os

# Keep OpenMP/BLAS single-threaded before NumPy and PyTorch load.
# This is the same safeguard used by experiments/run_benchmark.py.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import subprocess
import sys

from analysis.router_summary import (
    summarize_router_sensitivity,
)


def run_router_sensitivity():
    """
    Run the deterministic router sensitivity experiment.

    Returns
    -------
    dict
        Status information.
    """

    command = [
        sys.executable,
        "-m",
        "experiments.router_sensitivity",
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if process.returncode != 0:

        return {
            "status": "failed",
            "return_code": process.returncode,
            "stderr": process.stderr[-4000:],
        }

    return {
        "status": "success",
        "message": (
            "Router sensitivity experiment completed."
        ),
        "output_files": [
            "results/router_sensitivity_cv_results.csv",
            "results/router_sensitivity_summary.csv",
            "results/router_winner_stability.csv",
            "results/router_sharpness_summary.csv",
        ],
    }


def summarize_router_results():
    """
    Load and summarize existing router
    sensitivity experiment outputs.
    """

    summary = summarize_router_sensitivity(
        results_dir="results"
    )

    return summary


def execute_tool(
    tool_name,
    arguments,
):
    """
    Dispatch function calls requested by the LLM.
    """

    if tool_name == "run_router_sensitivity":

        return run_router_sensitivity()


    if tool_name == "summarize_router_results":

        return summarize_router_results()


    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# =========================================================
# Deterministic forecasting tools
#
# These functions wrap the existing forecasting engine.
# They compute diagnostics, candidates, validation scores,
# hard-routing choices, and forecast values.
#
# The LLM is not allowed to invent these numbers.
# Agent orchestration lives in agent/multi_agent.py.
# =========================================================


def _json_safe(value):
    """Convert NumPy values into JSON-friendly Python values."""

    import numpy as np

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return _json_safe(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return _json_safe(
            value.item()
        )

    return value


def diagnose_series(y):
    """
    Deterministic time-series diagnostics.

    Wraps diagnostics.basic.basic_diagnostics.
    """

    from diagnostics.basic import (
        basic_diagnostics,
    )

    return _json_safe(
        basic_diagnostics(y)
    )


def get_candidate_models(
    diagnostics,
    horizon,
    n_series=1,
    has_covariates=False,
    compute_budget="medium",
):
    """
    Deterministic candidate routing.

    Wraps routing.candidate_router.route_candidates.
    """

    from routing.candidate_router import (
        route_candidates,
    )

    candidates, reasons = route_candidates(
        diagnostics=diagnostics,
        horizon=horizon,
        n_series=n_series,
        has_covariates=has_covariates,
        compute_budget=compute_budget,
    )

    return {
        "candidates": list(candidates),
        "reasons": _json_safe(reasons),
    }


def run_validation(
    y,
    diagnostics,
    candidates,
    horizon,
    n_windows=1,
    window_types=None,
    train_window=60,
    fast=True,
    n_series=1,
    verbose=False,
):
    """
    Deterministic rolling-origin validation.

    Wraps evaluation.rolling_cv.rolling_backtest and the
    same per-expert configuration used by the benchmark.
    A failed expert is recorded and skipped, matching
    experiments/run_benchmark.py. If every expert fails,
    this function raises.
    """

    import pandas as pd

    from evaluation.rolling_cv import (
        rolling_backtest,
    )
    from models.config_builder import (
        build_model_kwargs,
    )
    from models.registry import (
        MODEL_REGISTRY,
        get_model_class,
    )

    if window_types is None:
        window_types = [
            "expanding",
            "sliding",
        ]

    y = (
        pd.Series(y)
        .dropna()
        .astype(float)
        .reset_index(drop=True)
    )

    seasonal_period = diagnostics.get(
        "seasonal_period"
    )

    inner_results = []
    failed_models = []

    for model_name in candidates:

        model_info = MODEL_REGISTRY[
            model_name
        ]

        if not model_info["enabled"]:
            continue

        if verbose:
            print(
                f"  Inner CV: {model_name}"
            )

        try:

            model_class = get_model_class(
                model_name
            )

            model_kwargs = build_model_kwargs(
                model_name=model_name,
                diagnostics=diagnostics,
                fast=fast,
                horizon=horizon,
                n_series=n_series,
            )

            for window_type in window_types:

                kwargs = {
                    "model_class": model_class,
                    "y": y,
                    "horizon": horizon,
                    "n_windows": n_windows,
                    "model_kwargs": model_kwargs,
                    "window_type": window_type,
                    "seasonal_period": seasonal_period,
                }

                if window_type == "sliding":
                    kwargs["train_window"] = (
                        train_window
                    )

                result = rolling_backtest(
                    **kwargs
                )

                result["model"] = model_name
                result["family"] = model_info[
                    "family"
                ]

                inner_results.append(
                    result
                )

        except Exception as exc:

            failed_models.append({
                "model": model_name,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            })

            print(
                f"    FAILED {model_name}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not inner_results:

        raise RuntimeError(
            "All candidate experts failed."
        )

    inner_results = pd.concat(
        inner_results,
        ignore_index=True,
    )

    validation_summary = (
        inner_results
        .groupby(
            [
                "family",
                "model",
                "window_type",
            ]
        )[
            [
                "mae",
                "rmse",
                "runtime",
            ]
        ]
        .mean()
        .reset_index()
    )

    return {
        "validation_summary": _json_safe(
            validation_summary.to_dict(
                orient="records"
            )
        ),
        "failed_models": failed_models,
    }


def select_expert(
    validation_summary,
    compute_weight=0.10,
    temperature=1.0,
):
    """
    Deterministic dynamic hard expert routing.

    Wraps routing.validation_router.compute_routing_scores.
    The selected expert is the highest-utility row.
    Routing weights are computed in Python.
    """

    import pandas as pd

    from routing.validation_router import (
        compute_routing_scores,
    )

    summary = pd.DataFrame(
        validation_summary
    )

    routing_table = compute_routing_scores(
        validation_summary=summary,
        metric="mae",
        compute_weight=compute_weight,
        temperature=temperature,
    )

    winner = (
        routing_table
        .sort_values(
            "utility",
            ascending=False,
        )
        .iloc[0]
    )

    selected = {
        "model": str(winner["model"]),
        "window_type": str(
            winner["window_type"]
        ),
        "validation_mae": float(
            winner["mae"]
        ),
        "utility": float(
            winner["utility"]
        ),
    }

    if "runtime" in winner.index:
        selected["runtime"] = float(
            winner["runtime"]
        )

    return _json_safe(selected)


def generate_forecast(
    y,
    diagnostics,
    model_name,
    window_type,
    horizon,
    fast=True,
    n_series=1,
    train_window=60,
):
    """
    Deterministic final forecast.

    Uses the same fit/predict path as
    experiments/run_benchmark.py:fit_one_expert.
    That helper is not imported here because the
    benchmark module runs its CLI on import.
    """

    import numpy as np
    import pandas as pd

    from models.config_builder import (
        build_model_kwargs,
    )
    from models.registry import (
        get_model_class,
    )

    y = (
        pd.Series(y)
        .dropna()
        .astype(float)
        .reset_index(drop=True)
    )

    model_kwargs = build_model_kwargs(
        model_name=model_name,
        diagnostics=diagnostics,
        fast=fast,
        horizon=horizon,
        n_series=n_series,
    )

    if window_type == "sliding":

        expert_train = (
            y.iloc[-train_window:]
            .reset_index(drop=True)
        )

    else:

        expert_train = (
            y.copy()
            .reset_index(drop=True)
        )

    model = get_model_class(
        model_name
    )(
        **model_kwargs
    )

    model.fit(expert_train)

    forecast = np.asarray(
        model.predict(
            horizon=horizon
        ),
        dtype=float,
    )

    forecast = np.squeeze(forecast)

    if forecast.ndim != 1:

        raise ValueError(
            f"{model_name}: invalid forecast "
            f"shape {forecast.shape}"
        )

    if len(forecast) != horizon:

        raise ValueError(
            f"{model_name}: forecast length "
            f"{len(forecast)} does not match "
            f"horizon {horizon}."
        )

    return [
        float(value)
        for value in forecast
    ]


def run_full_forecasting_pipeline(
    y,
    horizon,
    n_series=1,
    has_covariates=False,
    compute_budget="low",
    n_windows=1,
    window_types=None,
    train_window=60,
    fast=True,
    compute_weight=0.10,
    temperature=1.0,
    known_seasonal_period=None,
    model_allowlist=None,
    verbose=False,
):
    """
    Run the deterministic forecasting workflow.

    Order:
        diagnostics
        candidate routing
        rolling-origin validation
        dynamic hard expert selection
        final forecast

    This is the tool the forecasting agent calls.
    Every number in the returned JSON comes from
    the Python functions above.
    """

    if verbose:
        print(
            "\n[Forecasting Tool] diagnose_series"
        )

    diagnostics = diagnose_series(y)

    # Same known-period rule as experiments/run_benchmark.py.
    if (
        diagnostics.get("seasonal_period") is None
        and known_seasonal_period is not None
    ):
        diagnostics["seasonal_period"] = int(
            known_seasonal_period
        )

    if verbose:
        print(
            "\n[Forecasting Tool] "
            "get_candidate_models"
        )

    routed = get_candidate_models(
        diagnostics=diagnostics,
        horizon=horizon,
        n_series=n_series,
        has_covariates=has_covariates,
        compute_budget=compute_budget,
    )

    candidates = routed["candidates"]

    if model_allowlist is not None:
        allowed = set(model_allowlist)
        candidates = [
            model_name
            for model_name in candidates
            if model_name in allowed
        ]

    if not candidates:

        raise RuntimeError(
            "Candidate router returned no models."
        )

    if verbose:
        print(
            "\n[Forecasting Tool] run_validation"
        )

    validation = run_validation(
        y=y,
        diagnostics=diagnostics,
        candidates=candidates,
        horizon=horizon,
        n_windows=n_windows,
        window_types=window_types,
        train_window=train_window,
        fast=fast,
        n_series=n_series,
        verbose=verbose,
    )

    if verbose:
        print(
            "\n[Forecasting Tool] select_expert"
        )

    selected = select_expert(
        validation_summary=validation[
            "validation_summary"
        ],
        compute_weight=compute_weight,
        temperature=temperature,
    )

    if verbose:
        print(
            "\n[Forecasting Tool] generate_forecast"
        )

    forecast = generate_forecast(
        y=y,
        diagnostics=diagnostics,
        model_name=selected["model"],
        window_type=selected["window_type"],
        horizon=horizon,
        fast=fast,
        n_series=n_series,
        train_window=train_window,
    )

    evidence = (
        f"Selected {selected['model']} with a "
        f"{selected['window_type']} window because "
        f"it has the highest routing utility "
        f"({selected['utility']:.4f}) on "
        f"rolling-origin validation "
        f"(MAE {selected['validation_mae']:.4f}). "
        f"Utility and the forecast were computed "
        f"by Python tools, not by the LLM."
    )

    return {
        "diagnostics": diagnostics,
        "candidates": candidates,
        "candidate_reasons": routed["reasons"],
        "validation_summary": validation[
            "validation_summary"
        ],
        "failed_models": validation[
            "failed_models"
        ],
        "selected_expert": selected,
        "forecast": forecast,
        "evidence": evidence,
    }