"""
Multi-agent orchestration.

User request
    -> Planner Agent        (LLM plan)
    -> Forecasting Agent    (Python tools)
    -> Critic Agent         (LLM review + Python checks)
    -> optional one retry
    -> final structured result

Numerical forecasting stays in agent/tools.py.
This file only orders the agents.
"""

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd


MAX_RETRIES = 1

DISPLAY_NAMES = {
    "seasonal_naive": "SeasonalNaive",
    "auto_arima": "AutoARIMA",
    "ets": "ETS",
    "xgboost": "XGBoost",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "timesnet": "TimesNet",
    "itransformer": "iTransformer",
    "chronos2": "Chronos2",
    "timesfm3": "TimesFM3",
    "moirai2": "Moirai2",
}


def run_multi_agent(
    user_request,
    y,
    **forecast_settings,
):
    """
    Run planner -> forecaster -> critic through LangGraph.

    If the critic returns RETRY, the forecasting
    node runs once more with that feedback.
    """

    from agent.graph import run_forecast_graph

    horizon = forecast_settings.pop("horizon", None)
    dataset_source = forecast_settings.pop(
        "dataset_source",
        "series",
    )
    target_column = forecast_settings.pop(
        "target_column",
        None,
    )

    state = run_forecast_graph(
        user_request=user_request,
        values=list(y),
        horizon=horizon,
        dataset_source=dataset_source,
        target_column=target_column,
        settings=forecast_settings,
    )

    return {
        "planner": state["plan"],
        "forecast": state["forecast_result"],
        "critic": state["critic_result"],
        "retries": int(
            state.get("retry_count") or 0
        ),
    }


def _display_model(model_name):
    return DISPLAY_NAMES.get(
        model_name,
        model_name,
    )


def print_concise(result):
    """Default CLI report."""

    forecast_result = result["forecast"]
    critic_result = result["critic"]
    diagnostics = forecast_result["diagnostics"]
    selected = forecast_result["selected_expert"]

    print("Planner")
    print("✓ Forecast request understood")
    print()
    print("Diagnostics")
    print(
        "✓ Observations: "
        f"{diagnostics.get('n_obs')}"
    )
    print(
        "✓ Seasonal period: "
        f"{diagnostics.get('seasonal_period')}"
    )
    print()
    print("Validation")

    for row in forecast_result["validation_summary"]:
        name = _display_model(row["model"])
        print(
            f"{name:<14}"
            f"{row['window_type']:<12}"
            f"{float(row['mae']):.2f}"
        )

    print()
    print("Router")
    print(
        "✓ Selected "
        f"{_display_model(selected['model'])}"
        f" ({selected['window_type']})"
    )
    print()
    print("Critic")
    print(f"✓ {critic_result['decision']}")
    print()
    print("Forecast")

    for index, value in enumerate(
        forecast_result["forecast"],
        start=1,
    ):
        print(
            f"h{index} {float(value):.4f}"
        )


def print_report(result):
    """Print the portfolio CLI report."""

    planner_result = result["planner"]
    forecast_result = result["forecast"]
    critic_result = result["critic"]
    selected = forecast_result[
        "selected_expert"
    ]

    print(
        "\n==================================="
    )
    print("Planner")
    print(
        "==================================="
    )
    print(
        json.dumps(
            planner_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n==================================="
    )
    print("Forecasting Agent")
    print(
        "==================================="
    )
    print(
        "Selected expert: "
        + _display_model(selected["model"])
    )
    print(
        "Validation MAE: "
        f"{selected['validation_mae']:.4f}"
    )
    print(
        "Evidence: "
        + forecast_result["evidence"]
    )

    print(
        "\n==================================="
    )
    print("Critic")
    print(
        "==================================="
    )
    print(
        "Decision: "
        + critic_result["decision"]
    )

    if critic_result["issues"]:
        print("Issues:")
        for issue in critic_result["issues"]:
            print(f"- {issue}")

    if critic_result["decision"] == "RETRY":
        print(
            "Recommended action: "
            + critic_result[
                "recommended_action"
            ]
        )

    print(
        "\n==================================="
    )
    print("Final Forecast")
    print(
        "==================================="
    )

    for index, value in enumerate(
        forecast_result["forecast"],
        start=1,
    ):
        print(
            f"h{index}: {float(value):.4f}"
        )


def _load_cli_series(args, root):
    """Load the demo series from a dataset name or a local CSV."""

    from benchmarks.configs import (
        get_benchmark_config,
    )
    from benchmarks.datasets import (
        load_dataset,
    )

    if args.csv and args.dataset:
        raise SystemExit(
            "Pass either --dataset or --csv."
        )

    if args.csv:
        if not args.target:
            raise SystemExit(
                "--csv requires --target."
            )

        frame = pd.read_csv(args.csv)
        if args.target not in frame.columns:
            raise SystemExit(
                f"Column '{args.target}' was not found."
            )

        values = pd.to_numeric(
            frame[args.target],
            errors="coerce",
        ).dropna()
        config = get_benchmark_config("default")
        return {
            "name": args.target,
            "y": values,
            "n_series": 1,
            "has_covariates": False,
            "known_seasonal_period": None,
            "config": config,
            "dataset_source": args.csv,
            "target_column": args.target,
        }

    dataset_name = args.dataset or "airpassengers"
    dataset = load_dataset(
        dataset_name=dataset_name,
        data_root=root / "data",
    )
    config = get_benchmark_config(dataset_name)
    return {
        "name": dataset_name,
        "y": dataset["y"],
        "n_series": dataset["n_series"],
        "has_covariates": dataset["has_covariates"],
        "known_seasonal_period": dataset[
            "metadata"
        ].get("known_seasonal_period"),
        "config": config,
        "dataset_source": dataset_name,
        "target_column": dataset["metadata"].get(
            "target"
        ),
    }


def main():
    """
    AirPassengers demo by default.

    fast=True and one rolling window keep this
    command cheaper than experiments/run_benchmark.py.
    compute_budget="medium" includes the pretrained
    Chronos2 and TimesFM3 candidates.
    """

    from utils.reproducibility import (
        configure_runtime,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run the planner, forecasting agent, "
            "and critic."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Registered dataset name. Default: airpassengers.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Local CSV path for the CLI.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column when --csv is used.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    args = parser.parse_args()

    configure_runtime(verbose=args.verbose)

    root = Path(__file__).resolve().parents[1]
    loaded = _load_cli_series(args, root)
    config = loaded["config"]

    user_request = f"""
Forecast the next {args.horizon} observations
of {loaded["name"]}.

Use diagnostics, candidate routing,
rolling-origin validation, and dynamic
hard expert selection before forecasting.

Do not invent forecast numbers.
"""

    result = run_multi_agent(
        user_request=user_request,
        y=loaded["y"],
        horizon=args.horizon,
        dataset_source=loaded["dataset_source"],
        target_column=loaded["target_column"],
        n_series=loaded["n_series"],
        has_covariates=loaded["has_covariates"],
        known_seasonal_period=loaded[
            "known_seasonal_period"
        ],
        compute_budget="medium",
        n_windows=1,
        window_types=config["window_types"],
        train_window=config["train_window"],
        fast=True,
        compute_weight=config["compute_weight"],
        temperature=config["temperature"],
        verbose=args.verbose,
    )

    if args.verbose:
        print_report(result)
    else:
        print_concise(result)


if __name__ == "__main__":
    main()
