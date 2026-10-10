"""
Nested rolling-origin evaluation.

Protocol, unchanged from experiments/run_benchmark.py:

    outer train
        -> diagnostics on the outer train only
        -> candidate filtering
        -> inner rolling-origin validation on the outer train
        -> compute-aware utility
        -> dynamic hard expert selection
        -> refit the selected expert on the outer train
        -> score the outer test block

The outer test block is not used to choose an expert.
Oracle analysis is a separate post-hoc script.

Ablation routers reuse the same outer splits and the
same cached outer forecasts. They are not the primary method.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import numpy as np
import pandas as pd

from diagnostics.basic import basic_diagnostics
from evaluation.rolling_cv import rolling_backtest
from models.config_builder import build_model_kwargs
from models.registry import (
    MODEL_REGISTRY,
    get_model_class,
    list_enabled_models,
)
from routing.candidate_router import route_candidates
from routing.soft_ensemble import weighted_forecast
from routing.state_horizon_router import (
    build_state_horizon_routing_table,
)
from routing.validation_router import compute_routing_scores
from utils.reproducibility import get_logger


LOG = get_logger("nested_evaluation")


# Same state-router constants as
# experiments/compare_routing_strategies.py.
# They are not part of the primary hard router.
STATE_BANDWIDTH = 2.0
RECENCY_DECAY = 0.1

PRIMARY_ROUTER = "dynamic_hard_router"

# The only soft-routing ablation used by the paper path.
# routing_weight is the full softmax from compute_routing_scores
# over every validated expert. Do not truncate to top-k.
SOFT_ROUTING = "full_softmax"

ROUTER_METHODS = {
    "dynamic_hard_router",
    "dynamic_hard_router_no_compute_penalty",
    "dynamic_hard_router_no_candidate_filter",
    "soft_overall_router",
    "horizon_hard_router",
    "state_soft_router",
}


def run_nested_evaluation(
    y,
    settings,
    methods=None,
    verbose=False,
):
    """
    Run the nested protocol and return data frames.

    Parameters
    ----------
    y : array-like
        Full observed series. Splits are taken from the end.
    settings : dict
        Resolved experiment config.
    methods : list[str] or None
        Router methods to score. Fixed experts are always scored.
        The default is the primary dynamic hard router only.
    """

    if methods is None:
        methods = [PRIMARY_ROUTER]

    methods = list(methods)
    unknown = [
        method
        for method in methods
        if method not in ROUTER_METHODS
    ]
    if unknown:
        raise ValueError(
            f"Unknown router methods: {unknown}"
        )

    y = (
        pd.Series(y)
        .astype(float)
        .reset_index(drop=True)
    )

    horizon = int(settings["horizon"])
    outer_windows = int(settings["outer_windows"])
    inner_windows = int(settings["inner_windows"])
    train_window = int(settings["train_window"])
    window_types = list(settings["window_types"])
    compute_budget = settings["compute_budget"]
    compute_weight = float(settings["compute_weight"])
    temperature = float(settings["temperature"])
    allowlist = settings.get("model_allowlist")
    fast = bool(settings["fast_model_config"])
    seed = int(settings["seed"])
    dataset_name = settings["dataset"]
    config_name = settings.get(
        "config_name",
        dataset_name,
    )
    n_series = int(settings.get("n_series", 1))
    has_covariates = bool(
        settings.get("has_covariates", False)
    )
    known_seasonal_period = settings.get(
        "known_seasonal_period"
    )

    if allowlist is not None:
        allowlist = list(allowlist)

    total_test_size = outer_windows * horizon
    initial_train_size = len(y) - total_test_size

    if initial_train_size <= 0:
        raise ValueError(
            "Not enough observations for "
            "requested benchmark."
        )

    want_unfiltered = (
        "dynamic_hard_router_no_candidate_filter"
        in methods
    )

    prediction_rows = []
    selection_rows = []
    runtime_rows = []
    failure_rows = []

    for outer_idx in range(outer_windows):

        outer_train_end = (
            initial_train_size
            + outer_idx * horizon
        )
        outer_test_end = outer_train_end + horizon

        outer_train = (
            y.iloc[:outer_train_end]
            .reset_index(drop=True)
        )
        outer_test = (
            y.iloc[outer_train_end:outer_test_end]
            .reset_index(drop=True)
        )
        actual = outer_test.to_numpy(dtype=float)

        # Diagnostics see only the outer training block.
        diagnostics = basic_diagnostics(outer_train)
        seasonal_period = diagnostics.get(
            "seasonal_period"
        )
        if (
            seasonal_period is None
            and known_seasonal_period is not None
        ):
            seasonal_period = known_seasonal_period

        candidates, _ = route_candidates(
            diagnostics=diagnostics,
            horizon=horizon,
            n_series=n_series,
            has_covariates=has_covariates,
            compute_budget=compute_budget,
        )

        if allowlist is not None:
            candidates = [
                model_name
                for model_name in candidates
                if model_name in allowlist
            ]

        if want_unfiltered:
            pool = list_enabled_models()
            if allowlist is not None:
                pool = [
                    model_name
                    for model_name in pool
                    if model_name in allowlist
                ]
            models_to_run = list(
                dict.fromkeys(
                    list(candidates) + list(pool)
                )
            )
        else:
            models_to_run = list(candidates)

        if verbose:
            LOG.info(
                f"Outer origin {outer_idx + 1}: "
                f"candidates={candidates}"
            )

        inner_frames = []

        for model_name in models_to_run:

            if not MODEL_REGISTRY[model_name]["enabled"]:
                continue

            if verbose:
                LOG.info(f"  Inner CV: {model_name}")

            try:
                model_class = get_model_class(model_name)
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
                        "y": outer_train,
                        "horizon": horizon,
                        "n_windows": inner_windows,
                        "model_kwargs": model_kwargs,
                        "window_type": window_type,
                        "seasonal_period": seasonal_period,
                    }
                    if window_type == "sliding":
                        kwargs["train_window"] = (
                            train_window
                        )

                    result = rolling_backtest(**kwargs)
                    result["model"] = model_name
                    result["family"] = MODEL_REGISTRY[
                        model_name
                    ]["family"]
                    inner_frames.append(result)

            except Exception as exc:
                failure_rows.append({
                    "outer_origin": outer_idx + 1,
                    "model": model_name,
                    "stage": "inner_validation",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })
                LOG.info(
                    f"  FAILED {model_name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not inner_frames:
            raise RuntimeError(
                "All candidate experts failed."
            )

        inner_results = pd.concat(
            inner_frames,
            ignore_index=True,
        )
        validation = _validation_summary(inner_results)

        filtered_inner = inner_results[
            inner_results["model"].isin(candidates)
        ]
        filtered_validation = validation[
            validation["model"].isin(candidates)
        ]

        if (
            PRIMARY_ROUTER in methods
            and filtered_validation.empty
        ):
            raise RuntimeError(
                "All candidate experts failed."
            )

        forecast_cache = {}
        fit_seconds = {}

        def get_forecast(model_name, window_type):
            key = (model_name, window_type)
            if key not in forecast_cache:
                started = time.perf_counter()
                forecast_cache[key] = _fit_one_expert(
                    model_name=model_name,
                    window_type=window_type,
                    train_y=outer_train,
                    diagnostics=diagnostics,
                    horizon=horizon,
                    train_window=train_window,
                    fast=fast,
                    n_series=n_series,
                )
                fit_seconds[key] = (
                    time.perf_counter() - started
                )
            return forecast_cache[key]

        for _, expert in validation.iterrows():
            model_name = expert["model"]
            window_type = expert["window_type"]
            expert_label = _expert_label(
                model_name,
                window_type,
            )

            try:
                forecast = get_forecast(
                    model_name,
                    window_type,
                )
            except Exception as exc:
                failure_rows.append({
                    "outer_origin": outer_idx + 1,
                    "model": expert_label,
                    "stage": "outer_forecast",
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })
                LOG.info(
                    f"  OUTER FAILED {expert_label}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue

            _append_point_rows(
                prediction_rows,
                method=expert_label,
                method_type="fixed",
                model_name=model_name,
                window_type=window_type,
                selected_expert=expert_label,
                family=expert["family"],
                validation_loss=float(expert["mae"]),
                validation_runtime=float(
                    expert["runtime"]
                ),
                utility=None,
                actual=actual,
                forecast=forecast,
                outer_origin=outer_idx + 1,
                notes="",
            )
            runtime_rows.append(
                _runtime_row(
                    outer_origin=outer_idx + 1,
                    method=expert_label,
                    model_name=model_name,
                    window_type=window_type,
                    validation_loss=float(
                        expert["mae"]
                    ),
                    validation_runtime=float(
                        expert["runtime"]
                    ),
                    outer_fit_seconds=fit_seconds[
                        (model_name, window_type)
                    ],
                    selected_expert=expert_label,
                )
            )

        decisions = _router_decisions(
            methods=methods,
            filtered_validation=filtered_validation,
            filtered_inner=filtered_inner,
            all_validation=validation,
            all_inner=inner_results,
            outer_train=outer_train,
            seasonal_period=seasonal_period,
            horizon=horizon,
            compute_weight=compute_weight,
            temperature=temperature,
            get_forecast=get_forecast,
        )

        reported = None

        for decision in decisions:
            forecast = decision["forecast"]
            _append_point_rows(
                prediction_rows,
                method=decision["method"],
                method_type="router",
                model_name=decision["model"],
                window_type=decision["window_type"],
                selected_expert=decision[
                    "selected_expert"
                ],
                family=decision.get("family"),
                validation_loss=decision[
                    "validation_loss"
                ],
                validation_runtime=decision[
                    "validation_runtime"
                ],
                utility=decision.get("utility"),
                actual=actual,
                forecast=forecast,
                outer_origin=outer_idx + 1,
                notes=decision.get("notes", ""),
                lead_models=decision.get(
                    "lead_models"
                ),
            )

            if decision.get("lead_selections"):
                for item in decision["lead_selections"]:
                    selection_rows.append({
                        "outer_origin": outer_idx + 1,
                        "method": decision["method"],
                        "horizon_step": item[
                            "horizon_step"
                        ],
                        "model": item["model"],
                        "window_type": item[
                            "window_type"
                        ],
                        "selected_expert": item[
                            "selected_expert"
                        ],
                        "validation_loss": item[
                            "validation_loss"
                        ],
                        "validation_mae": item[
                            "validation_loss"
                        ],
                        "utility": item.get("utility"),
                        "runtime": item.get("runtime"),
                        "notes": decision.get(
                            "notes",
                            "",
                        ),
                    })
            else:
                selection_rows.append({
                    "outer_origin": outer_idx + 1,
                    "method": decision["method"],
                    "horizon_step": None,
                    "model": decision["model"],
                    "window_type": decision[
                        "window_type"
                    ],
                    "selected_expert": decision[
                        "selected_expert"
                    ],
                    "validation_loss": decision[
                        "validation_loss"
                    ],
                    "validation_mae": decision[
                        "validation_loss"
                    ],
                    "utility": decision.get("utility"),
                    "runtime": decision[
                        "validation_runtime"
                    ],
                    "notes": decision.get("notes", ""),
                })

            member_seconds = 0.0
            for key in decision["fit_keys"]:
                member_seconds += fit_seconds[key]

            runtime_rows.append(
                _runtime_row(
                    outer_origin=outer_idx + 1,
                    method=decision["method"],
                    model_name=decision["model"],
                    window_type=decision["window_type"],
                    validation_loss=decision[
                        "validation_loss"
                    ],
                    validation_runtime=decision[
                        "validation_runtime"
                    ],
                    outer_fit_seconds=member_seconds,
                    selected_expert=decision[
                        "selected_expert"
                    ],
                )
            )

            if decision["method"] == PRIMARY_ROUTER:
                reported = decision["selected_expert"]

        if reported is None and decisions:
            reported = decisions[0]["selected_expert"]

        LOG.info(
            f"Outer origin {outer_idx + 1}"
            f"/{outer_windows}: {reported}"
        )

    predictions = _annotate(
        pd.DataFrame(prediction_rows),
        dataset_name,
        config_name,
        seed,
        horizon,
    )
    selections = _annotate(
        pd.DataFrame(selection_rows),
        dataset_name,
        config_name,
        seed,
        horizon,
    )
    runtime = _annotate(
        pd.DataFrame(runtime_rows),
        dataset_name,
        config_name,
        seed,
        horizon,
    )
    failures = _annotate(
        pd.DataFrame(failure_rows),
        dataset_name,
        config_name,
        seed,
        horizon,
    )
    model_results = summarize_methods(
        predictions,
        runtime,
        outer_windows,
    )
    summary = benchmark_summary(
        model_results,
        methods,
    )

    return {
        "predictions": predictions,
        "model_results": model_results,
        "router_selections": selections,
        "summary": summary,
        "runtime": runtime,
        "failures": failures,
        "initial_train_size": initial_train_size,
    }


def summarize_methods(
    predictions,
    runtime,
    outer_windows,
):
    """MAE and RMSE from outer-test point errors."""

    rows = []

    if predictions.empty:
        return pd.DataFrame(rows)

    runtime_mean = (
        runtime
        .groupby("method", as_index=False)[
            ["validation_runtime", "outer_fit_seconds"]
        ]
        .mean()
    )

    for method, group in predictions.groupby("method"):
        n_origins = int(
            group["outer_origin"].nunique()
        )
        matched = runtime_mean[
            runtime_mean["method"] == method
        ]
        rows.append({
            "dataset": group["dataset"].iloc[0],
            "config_name": group["config_name"].iloc[0],
            "seed": int(group["seed"].iloc[0]),
            "horizon": int(group["horizon"].iloc[0]),
            "method": method,
            "method_type": group["method_type"].iloc[0],
            "mae": float(group["abs_error"].mean()),
            "rmse": float(
                np.sqrt(group["sq_error"].mean())
            ),
            "n_forecasts": int(len(group)),
            "n_outer_origins": n_origins,
            "complete_coverage": bool(
                n_origins == outer_windows
            ),
            "validation_runtime": (
                _optional_float(
                    matched["validation_runtime"].iloc[0]
                )
                if len(matched)
                else None
            ),
            "outer_fit_seconds": (
                _optional_float(
                    matched["outer_fit_seconds"].iloc[0]
                )
                if len(matched)
                else None
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mae")
        .reset_index(drop=True)
    )


def benchmark_summary(model_results, methods):
    """
    Paper comparison table.

    Fixed experts need a forecast at every outer origin.
    Requested routers are included when their coverage
    is complete. This matches the previous benchmark,
    which always reported the hard router and only the
    fixed experts that finished every origin.
    """

    if model_results.empty:
        return model_results.copy()

    fixed = model_results[
        (model_results["method_type"] == "fixed")
        & (model_results["complete_coverage"])
    ]
    routers = model_results[
        (model_results["method_type"] == "router")
        & (model_results["method"].isin(methods))
        & (model_results["complete_coverage"])
    ]

    columns = [
        "dataset",
        "config_name",
        "seed",
        "horizon",
        "method",
        "method_type",
        "mae",
        "rmse",
        "n_forecasts",
        "n_outer_origins",
        "validation_runtime",
        "outer_fit_seconds",
    ]

    return (
        pd.concat([routers, fixed], ignore_index=True)
        .sort_values("mae")
        .reset_index(drop=True)[columns]
    )


def _router_decisions(
    methods,
    filtered_validation,
    filtered_inner,
    all_validation,
    all_inner,
    outer_train,
    seasonal_period,
    horizon,
    compute_weight,
    temperature,
    get_forecast,
):
    decisions = []

    if PRIMARY_ROUTER in methods:
        winner = _hard_winner(
            filtered_validation,
            compute_weight,
            temperature,
        )
        decisions.append(
            _single_expert_decision(
                PRIMARY_ROUTER,
                winner,
                get_forecast,
                notes=(
                    "Highest compute-aware utility "
                    "on inner validation."
                ),
            )
        )

    if "dynamic_hard_router_no_compute_penalty" in methods:
        winner = _hard_winner(
            filtered_validation,
            compute_weight=0.0,
            temperature=temperature,
        )
        decisions.append(
            _single_expert_decision(
                "dynamic_hard_router_no_compute_penalty",
                winner,
                get_forecast,
                notes=(
                    "Same hard rule with "
                    "compute_weight=0."
                ),
            )
        )

    if "dynamic_hard_router_no_candidate_filter" in methods:
        winner = _hard_winner(
            all_validation,
            compute_weight,
            temperature,
        )
        decisions.append(
            _single_expert_decision(
                "dynamic_hard_router_no_candidate_filter",
                winner,
                get_forecast,
                notes=(
                    "Hard rule on every validated "
                    "expert. Diagnostic candidate "
                    "filtering is not used. An "
                    "experiment allowlist still applies."
                ),
            )
        )

    if "soft_overall_router" in methods:
        decisions.append(
            _soft_overall_decision(
                filtered_validation,
                compute_weight,
                temperature,
                get_forecast,
            )
        )

    if "horizon_hard_router" in methods:
        decisions.append(
            _horizon_hard_decision(
                filtered_inner,
                horizon,
                get_forecast,
            )
        )

    if "state_soft_router" in methods:
        decisions.append(
            _state_soft_decision(
                filtered_inner,
                outer_train,
                seasonal_period,
                horizon,
                compute_weight,
                temperature,
                get_forecast,
            )
        )

    return decisions


def _single_expert_decision(
    method,
    winner,
    get_forecast,
    notes,
):
    model_name = str(winner["model"])
    window_type = str(winner["window_type"])
    forecast = get_forecast(model_name, window_type)
    label = _expert_label(model_name, window_type)

    return {
        "method": method,
        "model": model_name,
        "window_type": window_type,
        "family": (
            str(winner["family"])
            if "family" in winner.index
            else None
        ),
        "selected_expert": label,
        "validation_loss": float(winner["mae"]),
        "validation_runtime": float(winner["runtime"]),
        "utility": float(winner["utility"]),
        "forecast": forecast,
        "fit_keys": [(model_name, window_type)],
        "notes": notes,
    }


def _soft_overall_decision(
    validation,
    compute_weight,
    temperature,
    get_forecast,
):
    """
    Overall soft-routing ablation.

    Definition: full softmax, no top-k truncation.
    Weights are the routing_weight column from
    routing.validation_router.compute_routing_scores
    over every validated expert. The mixture uses
    routing.soft_ensemble.weighted_forecast.
    """

    table = compute_routing_scores(
        validation_summary=validation,
        metric="mae",
        compute_weight=compute_weight,
        temperature=temperature,
    )

    forecasts = []
    weights = []
    keys = []

    for _, row in table.iterrows():
        model_name = str(row["model"])
        window_type = str(row["window_type"])
        try:
            forecasts.append(
                get_forecast(model_name, window_type)
            )
        except Exception as exc:
            LOG.info(
                f"  OUTER FAILED "
                f"{_expert_label(model_name, window_type)}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        weights.append(float(row["routing_weight"]))
        keys.append((model_name, window_type))

    if not forecasts:
        raise RuntimeError(
            "Soft router has no outer forecasts."
        )

    forecast = weighted_forecast(
        np.vstack(forecasts),
        weights,
    )
    top = table.iloc[0]

    return {
        "method": "soft_overall_router",
        "model": "ensemble",
        "window_type": "ensemble",
        "family": None,
        "selected_expert": "weighted_ensemble",
        "validation_loss": float(top["mae"]),
        "validation_runtime": float(
            np.average(
                table["runtime"],
                weights=table["routing_weight"],
            )
        ),
        "utility": float(top["utility"]),
        "forecast": forecast,
        "fit_keys": keys,
        "notes": (
            f"Soft-routing definition: {SOFT_ROUTING}. "
            "Full softmax over every validated expert, "
            "with no top-k truncation. "
            "validation_mae is the inner MAE of the "
            "highest-utility expert, not an ensemble loss."
        ),
    }


def _horizon_hard_decision(
    inner_results,
    horizon,
    get_forecast,
):
    """
    Horizon-specific hard routing.

    For each lead, choose the expert with the lowest
    mean inner absolute error at that lead. This is the
    rule in experiments/nested_routing_evaluation.py.
    That module executes a benchmark on import, so the
    rule is applied here without importing it.
    """

    winners = _horizon_hard_winners(
        inner_results,
        horizon,
    )
    forecast = np.zeros(horizon, dtype=float)
    lead_models = []
    lead_selections = []
    keys = []

    for _, winner in winners.iterrows():
        step = int(winner["horizon"])
        model_name = str(winner["model"])
        window_type = str(winner["window_type"])
        expert_forecast = get_forecast(
            model_name,
            window_type,
        )
        forecast[step - 1] = expert_forecast[step - 1]
        label = _expert_label(model_name, window_type)
        keys.append((model_name, window_type))
        lead_models.append({
            "horizon_step": step,
            "model": model_name,
            "window_type": window_type,
        })
        lead_selections.append({
            "horizon_step": step,
            "model": model_name,
            "window_type": window_type,
            "selected_expert": label,
            "validation_loss": float(
                winner["validation_loss"]
            ),
            "utility": None,
            "runtime": None,
        })

    return {
        "method": "horizon_hard_router",
        "model": "horizon_specific",
        "window_type": "horizon_specific",
        "family": None,
        "selected_expert": "horizon_specific",
        "validation_loss": float(
            winners["validation_loss"].mean()
        ),
        "validation_runtime": None,
        "utility": None,
        "forecast": forecast,
        "fit_keys": list(dict.fromkeys(keys)),
        "lead_models": lead_models,
        "lead_selections": lead_selections,
        "notes": (
            "Each lead uses the inner-CV expert with "
            "the lowest mean absolute error at that lead. "
            "This selection does not use the outer test."
        ),
    }


def _state_soft_decision(
    inner_results,
    outer_train,
    seasonal_period,
    horizon,
    compute_weight,
    temperature,
    get_forecast,
):
    """
    State-aware soft routing ablation.

    Weights come from
    routing.state_horizon_router.build_state_horizon_routing_table.
    The current state is the outer training series.
    """

    table = build_state_horizon_routing_table(
        all_results=inner_results,
        current_y=outer_train,
        horizon=horizon,
        seasonal_period=seasonal_period,
        compute_weight=compute_weight,
        temperature=temperature,
        state_bandwidth=STATE_BANDWIDTH,
        recency_decay=RECENCY_DECAY,
    )

    forecast = np.zeros(horizon, dtype=float)
    keys = []
    lead_selections = []

    for step in range(1, horizon + 1):
        slice_table = table[
            table["horizon"] == step
        ]
        member_forecasts = []
        member_weights = []

        for _, row in slice_table.iterrows():
            model_name = str(row["model"])
            window_type = str(row["window_type"])
            try:
                member_forecasts.append(
                    get_forecast(model_name, window_type)
                )
            except Exception as exc:
                LOG.info(
                    f"  OUTER FAILED "
                    f"{_expert_label(model_name, window_type)}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            member_weights.append(
                float(row["routing_weight"])
            )
            keys.append((model_name, window_type))

        if not member_forecasts:
            raise RuntimeError(
                "State router has no outer forecasts "
                f"at lead {step}."
            )

        mixed = weighted_forecast(
            np.vstack(member_forecasts),
            member_weights,
        )
        forecast[step - 1] = mixed[step - 1]
        top = slice_table.sort_values(
            "routing_weight",
            ascending=False,
        ).iloc[0]
        lead_selections.append({
            "horizon_step": step,
            "model": str(top["model"]),
            "window_type": str(top["window_type"]),
            "selected_expert": "weighted_ensemble",
            "validation_loss": float(
                top["localized_loss"]
            ),
            "utility": float(top["utility"]),
            "runtime": float(top["runtime"]),
        })

    return {
        "method": "state_soft_router",
        "model": "ensemble",
        "window_type": "ensemble",
        "family": None,
        "selected_expert": "weighted_ensemble",
        "validation_loss": float(
            np.mean([
                item["validation_loss"]
                for item in lead_selections
            ])
        ),
        "validation_runtime": float(
            np.mean([
                item["runtime"]
                for item in lead_selections
            ])
        ),
        "utility": None,
        "forecast": forecast,
        "fit_keys": list(dict.fromkeys(keys)),
        "lead_selections": lead_selections,
        "notes": (
            "State-aware ablation. Each lead uses the "
            "full per-horizon softmax from "
            "build_state_horizon_routing_table, with no "
            "top-k truncation. Bandwidth and recency "
            "match the legacy compare_routing_strategies.py "
            "settings. This is not the overall soft router."
        ),
    }


def _hard_winner(
    validation_summary,
    compute_weight,
    temperature,
):
    """Highest-utility expert. Same call as run_benchmark.py."""

    if validation_summary.empty:
        raise RuntimeError(
            "No validation rows for hard routing."
        )

    routing_table = compute_routing_scores(
        validation_summary=validation_summary,
        metric="mae",
        compute_weight=compute_weight,
        temperature=temperature,
    )

    return (
        routing_table
        .sort_values("utility", ascending=False)
        .iloc[0]
    )


def _horizon_hard_winners(inner_results, horizon):
    winners = []

    for step in range(1, horizon + 1):
        loss_column = f"abs_error_h{step}"
        if loss_column not in inner_results.columns:
            raise ValueError(
                f"Missing column: {loss_column}"
            )

        table = (
            inner_results
            .groupby(
                ["family", "model", "window_type"],
                as_index=False,
            )[loss_column]
            .mean()
            .sort_values(loss_column, ascending=True)
            .reset_index(drop=True)
        )
        best = table.iloc[0]
        winners.append({
            "horizon": step,
            "family": best["family"],
            "model": best["model"],
            "window_type": best["window_type"],
            "validation_loss": float(best[loss_column]),
        })

    return pd.DataFrame(winners)


def _validation_summary(inner_results):
    """Mean inner-window MAE, RMSE, and runtime."""

    return (
        inner_results
        .groupby(
            ["family", "model", "window_type"],
            as_index=False,
        )[["mae", "rmse", "runtime"]]
        .mean()
    )


def _fit_one_expert(
    model_name,
    window_type,
    train_y,
    diagnostics,
    horizon,
    train_window,
    fast,
    n_series,
):
    """
    Refit one expert on the current outer train.

    Same fit and predict path as
    experiments/run_benchmark.py. That script is not
    imported because it used to execute on import.
    """

    model_kwargs = build_model_kwargs(
        model_name=model_name,
        diagnostics=diagnostics,
        fast=fast,
        horizon=horizon,
        n_series=n_series,
    )

    if window_type == "sliding":
        expert_train = (
            train_y.iloc[-train_window:]
            .reset_index(drop=True)
        )
    else:
        expert_train = (
            train_y.copy().reset_index(drop=True)
        )

    model = get_model_class(model_name)(**model_kwargs)
    model.fit(expert_train)

    forecast = np.squeeze(
        np.asarray(
            model.predict(horizon=horizon),
            dtype=float,
        )
    )

    if forecast.ndim != 1:
        raise ValueError(
            f"{model_name}: invalid forecast "
            f"shape {forecast.shape}"
        )

    if len(forecast) != horizon:
        raise ValueError(
            f"{model_name}: expected {horizon} "
            f"forecasts, received {len(forecast)}."
        )

    return forecast


def _append_point_rows(
    rows,
    method,
    method_type,
    model_name,
    window_type,
    selected_expert,
    family,
    validation_loss,
    validation_runtime,
    utility,
    actual,
    forecast,
    outer_origin,
    notes,
    lead_models=None,
):
    for step in range(len(actual)):
        error = float(actual[step] - forecast[step])
        lead_model = model_name
        lead_window = window_type
        lead_expert = selected_expert

        if lead_models is not None:
            lead_model = lead_models[step]["model"]
            lead_window = lead_models[step]["window_type"]
            lead_expert = _expert_label(
                lead_model,
                lead_window,
            )

        rows.append({
            "outer_origin": outer_origin,
            "horizon_step": step + 1,
            "method": method,
            "method_type": method_type,
            "family": family,
            "model": lead_model,
            "window_type": lead_window,
            "selected_expert": lead_expert,
            "validation_loss": validation_loss,
            "validation_mae": validation_loss,
            "validation_runtime": validation_runtime,
            "runtime": validation_runtime,
            "utility": utility,
            "actual": float(actual[step]),
            "prediction": float(forecast[step]),
            "error": error,
            "abs_error": float(abs(error)),
            "sq_error": float(error ** 2),
            "notes": notes,
        })


def _runtime_row(
    outer_origin,
    method,
    model_name,
    window_type,
    validation_loss,
    validation_runtime,
    outer_fit_seconds,
    selected_expert,
):
    return {
        "outer_origin": outer_origin,
        "method": method,
        "model": model_name,
        "window_type": window_type,
        "selected_expert": selected_expert,
        "validation_loss": validation_loss,
        "validation_mae": validation_loss,
        "validation_runtime": validation_runtime,
        "runtime": validation_runtime,
        "outer_fit_seconds": outer_fit_seconds,
    }


def _optional_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _expert_label(model_name, window_type):
    return f"{model_name}__{window_type}"


def _annotate(frame, dataset_name, config_name, seed, horizon):
    if frame.empty:
        return frame

    annotated = frame.copy()
    annotated.insert(0, "horizon", horizon)
    annotated.insert(0, "seed", seed)
    annotated.insert(0, "config_name", config_name)
    annotated.insert(0, "dataset", dataset_name)
    return annotated
