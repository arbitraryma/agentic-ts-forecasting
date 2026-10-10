"""
Ablation runner.

All ablation methods share one nested evaluation, so
they use the same outer origins and the same outer
forecast cache.

    python -m experiments.run_ablation --config configs/airpassengers.yaml

This does not call an LLM.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

from benchmarks.configs import resolve_experiment_config
from benchmarks.datasets import load_dataset
from evaluation.nested_evaluation import run_nested_evaluation
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


# Fixed experts are always scored by the evaluator.
# soft_overall_router uses the full softmax from
# evaluation.nested_evaluation.SOFT_ROUTING.
# It does not truncate to a top-k subset.
ABLATION_ROUTERS = [
    "dynamic_hard_router",
    "dynamic_hard_router_no_compute_penalty",
    "dynamic_hard_router_no_candidate_filter",
    "soft_overall_router",
    "horizon_hard_router",
    "state_soft_router",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Ablations of validation-driven routing "
            "on shared outer splits."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--outer-windows", type=int, default=None)
    parser.add_argument("--inner-windows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
    config["experiment"] = "ablation"

    logger = get_logger("ablation")
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

    emit("Ablation")
    emit(
        f"dataset={config['dataset']}  "
        f"horizon={config['horizon']}  "
        f"outer={config['outer_windows']}  "
        f"inner={config['inner_windows']}  "
        f"seed={config['seed']}"
    )
    emit(
        "Methods: fixed experts, "
        + ", ".join(ABLATION_ROUTERS)
    )

    result = run_nested_evaluation(
        y=dataset["y"],
        settings=config,
        methods=ABLATION_ROUTERS,
        verbose=args.verbose,
    )

    directory = create_experiment_dir(
        experiment_name="ablation",
        config=config,
    )
    save_config(directory, config)
    save_frame(directory, "predictions.csv", result["predictions"])
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
    save_frame(directory, "summary.csv", result["summary"])
    save_frame(directory, "runtime.csv", result["runtime"])

    if not result["failures"].empty:
        save_frame(
            directory,
            "failures.csv",
            result["failures"],
        )

    emit("")
    emit(
        result["summary"][
            ["method", "method_type", "mae", "rmse"]
        ].to_string(index=False)
    )
    emit("")
    emit(f"Saved: {directory}")
    save_text_log(directory, lines)
    return directory


if __name__ == "__main__":
    main()
