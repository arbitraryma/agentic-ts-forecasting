"""
Config-driven nested forecast benchmark.

The numerical protocol lives in
evaluation/nested_evaluation.py. This runner loads
a YAML config, scores fixed experts and the dynamic
hard router, and writes a new results directory.

It does not call the multi-agent layer and does not
need an LLM API key.

    python -m experiments.run_benchmark --config configs/airpassengers.yaml

The previous flags still work:

    python -m experiments.run_benchmark --dataset airpassengers --fast
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

from benchmarks.configs import resolve_experiment_config
from benchmarks.datasets import load_dataset
from evaluation.nested_evaluation import (
    PRIMARY_ROUTER,
    run_nested_evaluation,
)
from utils.experiment_logging import (
    create_experiment_dir,
    save_config,
    save_frame,
    save_text_log,
)
from utils.reproducibility import (
    configure_runtime,
    get_logger,
    set_global_seed,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Nested rolling benchmark for "
            "validation-driven dynamic hard routing."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config, for example configs/airpassengers.yaml.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. Used when --config is omitted.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )
    parser.add_argument("--outer-windows", type=int, default=None)
    parser.add_argument("--inner-windows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model allowlist.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the existing fast model settings.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-model validation progress.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_runtime(verbose=args.verbose)

    overrides = {
        "outer_windows": args.outer_windows,
        "inner_windows": args.inner_windows,
        "horizon": args.horizon,
        "seed": args.seed,
    }
    if args.fast:
        overrides["fast_model_config"] = True
    if args.models:
        overrides["model_allowlist"] = [
            item.strip()
            for item in args.models.split(",")
            if item.strip()
        ]

    config = resolve_experiment_config(
        config_path=args.config,
        dataset=args.dataset,
        overrides=overrides,
    )

    logger = get_logger("benchmark")
    lines = []

    def emit(message):
        logger.info(message)
        lines.append(message)

    set_global_seed(config["seed"])

    dataset = load_dataset(
        dataset_name=config["dataset"],
        data_root=args.data_root,
        target=config.get("target"),
    )

    config["n_series"] = dataset["n_series"]
    config["has_covariates"] = dataset["has_covariates"]
    config["known_seasonal_period"] = dataset[
        "known_seasonal_period"
    ]
    config["frequency"] = dataset["frequency"]

    emit("Benchmark")
    emit(
        f"dataset={config['dataset']}  "
        f"horizon={config['horizon']}  "
        f"outer={config['outer_windows']}  "
        f"inner={config['inner_windows']}  "
        f"seed={config['seed']}"
    )
    emit(f"fast_model_config={config['fast_model_config']}")

    result = run_nested_evaluation(
        y=dataset["y"],
        settings=config,
        methods=[PRIMARY_ROUTER],
        verbose=args.verbose,
    )

    directory = create_experiment_dir(
        experiment_name="benchmark",
        config=config,
    )
    save_config(directory, config)
    save_frame(
        directory,
        "predictions.csv",
        result["predictions"],
    )
    save_frame(
        directory,
        "model_results.csv",
        result["model_results"],
    )
    save_frame(
        directory,
        "router_selections.csv",
        result["router_selections"],
    )
    save_frame(
        directory,
        "summary.csv",
        result["summary"],
    )
    save_frame(
        directory,
        "runtime.csv",
        result["runtime"],
    )

    if not result["failures"].empty:
        save_frame(
            directory,
            "failures.csv",
            result["failures"],
        )

    # Previous benchmark filenames, kept inside this
    # run directory so older readers can find them
    # without overwriting results/benchmarks/.
    predictions = result["predictions"]
    save_frame(
        directory,
        "router_predictions.csv",
        predictions[
            predictions["method"] == PRIMARY_ROUTER
        ],
    )
    save_frame(
        directory,
        "fixed_expert_predictions.csv",
        predictions[
            predictions["method_type"] == "fixed"
        ],
    )
    save_frame(
        directory,
        "comparison.csv",
        result["summary"],
    )

    emit("")
    emit(
        result["summary"][
            ["method", "mae", "rmse", "n_forecasts"]
        ].to_string(index=False)
    )
    emit("")
    emit("Router selections")
    emit(
        result["router_selections"][
            [
                "outer_origin",
                "selected_expert",
                "validation_loss",
                "utility",
            ]
        ].to_string(index=False)
    )
    emit("")
    emit(f"Saved: {directory}")
    save_text_log(directory, lines)
    return directory


if __name__ == "__main__":
    main()
