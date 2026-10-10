"""
Paper figures from one saved benchmark directory.

    python -m analysis.make_figures --results <experiment_dir>

Figures are written to results/figures/ and are not
replaced when a file with the same name already exists.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.oracle_analysis import oracle_analysis


PRIMARY_ROUTER = "dynamic_hard_router"


def save_figure(figure, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _unused(output_dir / f"{stem}.png")
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {path}")
    return path


def plot_selection_frequency(selections, output_dir, stem):
    primary = selections[
        selections["method"] == PRIMARY_ROUTER
    ]
    counts = (
        primary["selected_expert"]
        .value_counts()
        .sort_values(ascending=True)
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.barh(counts.index, counts.values)
    axis.set_xlabel("Outer origins")
    axis.set_title("Dynamic hard router selection frequency")
    figure.tight_layout()
    return save_figure(figure, output_dir, stem)


def plot_selected_expert_by_origin(selections, output_dir, stem):
    primary = (
        selections[selections["method"] == PRIMARY_ROUTER]
        .sort_values("outer_origin")
    )
    labels = pd.Categorical(
        primary["selected_expert"].tolist()
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(
        primary["outer_origin"],
        labels.codes,
    )
    axis.set_yticks(range(len(labels.categories)))
    axis.set_yticklabels(list(labels.categories))
    axis.set_xlabel("Outer origin")
    axis.set_title("Selected expert by outer origin")
    figure.tight_layout()
    return save_figure(figure, output_dir, stem)


def plot_mae_comparison(summary, output_dir, stem):
    ordered = summary.sort_values("mae", ascending=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = [
        "#1f4e79"
        if method == PRIMARY_ROUTER
        else "#9aa5b1"
        for method in ordered["method"]
    ]
    axis.barh(ordered["method"], ordered["mae"], color=colors)
    axis.set_xlabel("Outer-test MAE")
    axis.set_title("Fixed experts and dynamic hard router")
    figure.tight_layout()
    return save_figure(figure, output_dir, stem)


def plot_runtime_vs_mae(model_results, output_dir, stem):
    frame = model_results.dropna(
        subset=["outer_fit_seconds", "mae"]
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        frame["outer_fit_seconds"],
        frame["mae"],
    )
    for _, row in frame.iterrows():
        axis.annotate(
            row["method"],
            (
                row["outer_fit_seconds"],
                row["mae"],
            ),
            fontsize=7,
        )
    axis.set_xlabel("Mean outer refit seconds")
    axis.set_ylabel("Outer-test MAE")
    axis.set_title("Runtime versus MAE")
    figure.tight_layout()
    return save_figure(figure, output_dir, stem)


def plot_oracle_regret(by_origin, output_dir, stem):
    ordered = by_origin.sort_values("outer_origin")
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(
        ordered["outer_origin"].astype(str),
        ordered["origin_regret"],
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Outer origin")
    axis.set_ylabel("Router MAE minus oracle MAE")
    axis.set_title(
        "Post-hoc oracle regret by origin"
    )
    figure.tight_layout()
    return save_figure(figure, output_dir, stem)


def make_figures(results_dir, output_dir):
    results_dir = Path(results_dir)
    prefix = results_dir.name
    selections = pd.read_csv(
        results_dir / "router_selections.csv"
    )
    summary = pd.read_csv(results_dir / "summary.csv")
    model_results = pd.read_csv(
        results_dir / "model_results.csv"
    )
    predictions = pd.read_csv(
        results_dir / "predictions.csv"
    )
    oracle = oracle_analysis(predictions)

    plot_selection_frequency(
        selections,
        output_dir,
        f"{prefix}_selection_frequency",
    )
    plot_selected_expert_by_origin(
        selections,
        output_dir,
        f"{prefix}_selected_expert_by_origin",
    )
    plot_mae_comparison(
        summary,
        output_dir,
        f"{prefix}_mae_comparison",
    )
    plot_runtime_vs_mae(
        model_results,
        output_dir,
        f"{prefix}_runtime_vs_mae",
    )
    plot_oracle_regret(
        oracle["by_origin"],
        output_dir,
        f"{prefix}_oracle_regret",
    )


def _unused(path):
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(
            f"{path.stem}_{index}{path.suffix}"
        )
        if not candidate.exists():
            return candidate
        index += 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write paper figures for one saved run."
    )
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures",
    )
    args = parser.parse_args(argv)
    make_figures(args.results, args.output)


if __name__ == "__main__":
    main()
