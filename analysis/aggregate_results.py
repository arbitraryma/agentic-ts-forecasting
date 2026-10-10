"""
Combine saved benchmark summaries across datasets.

    python -m analysis.aggregate_results --results results/benchmark

Improvement is a difference of MAE on the saved
outer-test errors. It is not a significance claim.
"""

import argparse
from pathlib import Path

import pandas as pd


FOUNDATION_MODELS = (
    "chronos2",
    "timesfm3",
    "moirai2",
)

PRIMARY_ROUTER = "dynamic_hard_router"


def find_summaries(results_root):
    root = Path(results_root)
    paths = sorted(root.glob("**/summary.csv"))
    return [
        path
        for path in paths
        if "tables" not in path.parts
    ]


def aggregate_summaries(frames):
    """
    Build per-dataset scores, average rank, and
    router improvements.
    """

    if not frames:
        raise ValueError("No summary.csv files were found.")

    combined = pd.concat(frames, ignore_index=True)

    per_dataset = (
        combined[
            [
                "dataset",
                "config_name",
                "method",
                "method_type",
                "mae",
                "rmse",
                "outer_fit_seconds",
                "validation_runtime",
            ]
        ]
        .copy()
    )

    rank_rows = []
    for dataset, group in per_dataset.groupby("dataset"):
        ordered = group.sort_values("mae").reset_index(drop=True)
        ordered["rank"] = np_rank(ordered)
        for _, row in ordered.iterrows():
            rank_rows.append({
                "dataset": dataset,
                "method": row["method"],
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "rank": int(row["rank"]),
            })

    ranks = pd.DataFrame(rank_rows)
    average_rank = (
        ranks
        .groupby("method", as_index=False)
        .agg(
            average_rank=("rank", "mean"),
            n_datasets=("dataset", "nunique"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values("average_rank")
        .reset_index(drop=True)
    )

    improvements = []
    for dataset, group in per_dataset.groupby("dataset"):
        router = group[group["method"] == PRIMARY_ROUTER]
        fixed = group[group["method_type"] == "fixed"]
        if router.empty or fixed.empty:
            continue

        router_mae = float(router["mae"].iloc[0])
        best_fixed = fixed.sort_values("mae").iloc[0]
        foundation = fixed[
            fixed["method"].str.startswith(FOUNDATION_MODELS)
        ]
        row = {
            "dataset": dataset,
            "router_mae": router_mae,
            "best_fixed_expert": best_fixed["method"],
            "best_fixed_mae": float(best_fixed["mae"]),
            "improvement_over_best_fixed_mae": (
                float(best_fixed["mae"]) - router_mae
            ),
            "strongest_foundation_model": None,
            "strongest_foundation_mae": None,
            "improvement_over_strongest_foundation_mae": None,
            "n_datasets_in_this_row": 1,
            "note": (
                "Positive improvement means the router "
                "MAE is lower. One dataset is not a "
                "general result."
            ),
        }
        if not foundation.empty:
            best_foundation = foundation.sort_values(
                "mae"
            ).iloc[0]
            row["strongest_foundation_model"] = (
                best_foundation["method"]
            )
            row["strongest_foundation_mae"] = float(
                best_foundation["mae"]
            )
            row[
                "improvement_over_strongest_foundation_mae"
            ] = (
                float(best_foundation["mae"]) - router_mae
            )
        improvements.append(row)

    runtime = (
        per_dataset[
            [
                "dataset",
                "method",
                "method_type",
                "mae",
                "outer_fit_seconds",
                "validation_runtime",
            ]
        ]
        .sort_values(["dataset", "mae"])
        .reset_index(drop=True)
    )

    return {
        "per_dataset": per_dataset,
        "average_rank": average_rank,
        "improvements": pd.DataFrame(improvements),
        "runtime": runtime,
    }


def np_rank(ordered):
    # Competition rank on the already sorted MAE column.
    ranks = []
    previous = None
    rank = 0
    for index, value in enumerate(ordered["mae"].tolist(), start=1):
        if previous is None or value != previous:
            rank = index
        ranks.append(rank)
        previous = value
    return ranks


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate saved benchmark summaries."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results/benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/aggregate",
    )
    args = parser.parse_args(argv)

    frames = []
    for path in find_summaries(args.results):
        frame = pd.read_csv(path)
        frame["source"] = str(path)
        frames.append(frame)

    tables = aggregate_summaries(frames)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for name, frame in tables.items():
        destination = _unused_path(
            output / f"{name}.csv"
        )
        frame.to_csv(destination, index=False)
        print(f"Saved: {destination}")

    if not tables["improvements"].empty:
        print(
            tables["improvements"].to_string(index=False)
        )


def _unused_path(path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


if __name__ == "__main__":
    main()
