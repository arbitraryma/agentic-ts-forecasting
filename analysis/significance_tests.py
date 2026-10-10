"""
Paired comparisons across outer origins.

A small outer-origin count is reported, not treated
as evidence of a general effect.

    python -m analysis.significance_tests --results <experiment_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# Below this count, a p-value is computed but is not
# a basis for a significance claim.
MINIMUM_ORIGINS_FOR_INTERPRETATION = 10


def origin_mae(predictions, method):
    group = predictions[predictions["method"] == method]
    if group.empty:
        raise ValueError(f"Method not found: {method}")
    return (
        group
        .groupby("outer_origin")["abs_error"]
        .mean()
        .sort_index()
    )


def paired_origin_test(
    predictions,
    method_a,
    method_b,
    seed=42,
    n_bootstrap=2000,
):
    """
    Compare two methods on shared outer origins.

    The difference is method_a minus method_b, so a
    negative mean favors method_a when the metric is MAE.
    """

    left = origin_mae(predictions, method_a)
    right = origin_mae(predictions, method_b)
    common = left.index.intersection(right.index)

    differences = (
        left.loc[common].to_numpy(dtype=float)
        - right.loc[common].to_numpy(dtype=float)
    )
    n_origins = int(len(differences))
    mean_difference = (
        float(differences.mean())
        if n_origins
        else None
    )

    interpretable = (
        n_origins >= MINIMUM_ORIGINS_FOR_INTERPRETATION
    )
    warning = None
    if not interpretable:
        warning = (
            f"{n_origins} outer origins is below "
            f"{MINIMUM_ORIGINS_FOR_INTERPRETATION}. "
            "Do not report this comparison as "
            "statistically significant."
        )

    statistic = None
    p_value = None
    nonzero = differences[differences != 0]
    if len(nonzero) >= 5:
        result = wilcoxon(
            nonzero,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    elif n_origins > 0:
        warning = (
            (warning + " ") if warning else ""
        ) + (
            "Wilcoxon was not computed because fewer "
            "than 5 origins have a non-zero difference."
        )

    lower = None
    upper = None
    if n_origins >= 2:
        generator = np.random.default_rng(int(seed))
        draws = generator.integers(
            0,
            n_origins,
            size=(n_bootstrap, n_origins),
        )
        samples = differences[draws].mean(axis=1)
        lower = float(np.quantile(samples, 0.025))
        upper = float(np.quantile(samples, 0.975))

    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_outer_origins": n_origins,
        "mean_paired_mae_difference": mean_difference,
        "bootstrap_ci_low": lower,
        "bootstrap_ci_high": upper,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": p_value,
        "interpretable": interpretable,
        "warning": warning,
    }


def compare_router_with_fixed(
    predictions,
    router_method="dynamic_hard_router",
    seed=42,
):
    fixed_methods = sorted(
        predictions.loc[
            predictions["method_type"] == "fixed",
            "method",
        ].unique()
    )
    rows = [
        paired_origin_test(
            predictions,
            router_method,
            method,
            seed=seed,
        )
        for method in fixed_methods
    ]
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Paired outer-origin comparisons. "
            "Small samples are flagged."
        )
    )
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--router",
        type=str,
        default="dynamic_hard_router",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    directory = Path(args.results)
    predictions = pd.read_csv(
        directory / "predictions.csv"
    )
    table = compare_router_with_fixed(
        predictions,
        router_method=args.router,
        seed=args.seed,
    )
    destination = directory / "significance_tests.csv"
    table.to_csv(destination, index=False)
    print(table.to_string(index=False))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
