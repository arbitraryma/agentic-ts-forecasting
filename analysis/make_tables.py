"""
Paper tables from saved experiments and the dataset registry.

    python -m analysis.make_tables --benchmark <benchmark_dir>

Missing ablation or benchmark files are reported.
Numbers are not invented.
"""

import argparse
from pathlib import Path

import pandas as pd

from analysis.oracle_analysis import oracle_analysis
from benchmarks.datasets import load_dataset
from benchmarks.registry import DATASET_REGISTRY


def dataset_characteristics(data_root="data"):
    rows = []
    for name, spec in DATASET_REGISTRY.items():
        n_obs = None
        if spec["status"] == "available":
            loaded = load_dataset(
                name,
                data_root=data_root,
                target=spec.get("target"),
            )
            n_obs = int(len(loaded["y"]))
        rows.append({
            "dataset": name,
            "status": spec["status"],
            "frequency": spec["frequency"],
            "n_obs": n_obs,
            "n_series": spec["n_series"],
            "known_seasonal_period": spec[
                "known_seasonal_period"
            ],
            "target": spec["target"],
            "has_covariates": spec["has_covariates"],
            "description": spec["description"],
        })
    return pd.DataFrame(rows)


def main_benchmark_table(summary):
    columns = [
        "dataset",
        "method",
        "method_type",
        "mae",
        "rmse",
        "n_forecasts",
        "n_outer_origins",
    ]
    present = [
        column
        for column in columns
        if column in summary.columns
    ]
    return (
        summary[present]
        .sort_values(["dataset", "mae"])
        .reset_index(drop=True)
    )


def ablation_table(summary):
    return main_benchmark_table(summary)


def runtime_table(runtime):
    grouped = (
        runtime
        .groupby(
            ["dataset", "method"],
            as_index=False,
        )
        .agg(
            mean_validation_runtime=(
                "validation_runtime",
                "mean",
            ),
            mean_outer_fit_seconds=(
                "outer_fit_seconds",
                "mean",
            ),
            n_outer_origins=("outer_origin", "nunique"),
        )
    )
    return grouped.sort_values(
        ["dataset", "mean_outer_fit_seconds"]
    ).reset_index(drop=True)


def oracle_table(predictions):
    tables = oracle_analysis(predictions)
    summary = tables["summary"].copy()
    summary["post_hoc"] = True
    return summary


def write_table(frame, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _unused(output_dir / f"{stem}.csv")
    tex_path = _unused(
        csv_path.with_suffix(".tex")
    )
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        to_latex(frame, stem),
        encoding="utf-8",
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


def to_latex(frame, caption):
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else f"{value:.4f}"
                )
            )
    columns = [
        _escape(str(column))
        for column in display.columns
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{" + ("l" * len(columns)) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for _, row in display.iterrows():
        cells = [
            _escape("" if pd.isna(value) else str(value))
            for value in row.tolist()
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{" + _escape(caption) + "}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def _escape(text):
    return (
        text
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
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
        description="Write CSV and LaTeX paper tables."
    )
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="results/tables",
    )
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args(argv)

    write_table(
        dataset_characteristics(args.data_root),
        args.output,
        "table1_dataset_characteristics",
    )

    if args.benchmark is None:
        print(
            "No benchmark directory given. "
            "Tables 2, 4, and 5 were not written."
        )
        return

    benchmark_dir = Path(args.benchmark)
    summary = pd.read_csv(benchmark_dir / "summary.csv")
    runtime = pd.read_csv(benchmark_dir / "runtime.csv")
    predictions = pd.read_csv(
        benchmark_dir / "predictions.csv"
    )

    write_table(
        main_benchmark_table(summary),
        args.output,
        "table2_main_benchmark",
    )
    write_table(
        runtime_table(runtime),
        args.output,
        "table4_runtime",
    )
    write_table(
        oracle_table(predictions),
        args.output,
        "table5_oracle_diagnostic",
    )

    if args.ablation is None:
        print(
            "No ablation directory given. "
            "Table 3 was not written."
        )
        return

    ablation = pd.read_csv(
        Path(args.ablation) / "summary.csv"
    )
    write_table(
        ablation_table(ablation),
        args.output,
        "table3_ablation",
    )


if __name__ == "__main__":
    main()
