"""
Post-hoc oracle diagnostics.

These calculations use the outer test after routing
has already been recorded. They must not be fed back
into expert selection.

    python -m analysis.oracle_analysis --results <experiment_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


POST_HOC_NOTE = (
    "Post-hoc diagnostic. The outer test was not "
    "available to the router."
)


def oracle_analysis(
    predictions,
    router_method="dynamic_hard_router",
):
    """
    Compare one router with the ex-post best fixed expert.

    Oracle MAE is the pooled absolute error of the fixed
    expert that was best on each outer origin after the
    fact. Router regret is router MAE minus that oracle
    MAE. Selection accuracy is the fraction of origins
    where the selected expert is one of the origin's
    minimizers.
    """

    required = {
        "method",
        "method_type",
        "outer_origin",
        "horizon_step",
        "abs_error",
        "selected_expert",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Predictions are missing columns: {sorted(missing)}"
        )

    fixed = predictions[
        predictions["method_type"] == "fixed"
    ].copy()
    router = predictions[
        predictions["method"] == router_method
    ].copy()

    if fixed.empty or router.empty:
        raise ValueError(
            "Oracle analysis needs fixed experts and "
            f"{router_method} predictions."
        )

    fixed_origin = (
        fixed
        .groupby(
            ["outer_origin", "method"],
            as_index=False,
        )["abs_error"]
        .mean()
        .rename(columns={"abs_error": "origin_mae"})
    )

    best_mae = (
        fixed_origin
        .groupby("outer_origin", as_index=False)["origin_mae"]
        .min()
        .rename(columns={"origin_mae": "oracle_origin_mae"})
    )

    origin_rows = []
    oracle_point_rows = []

    for origin, group in fixed_origin.groupby("outer_origin"):
        best_value = float(group["origin_mae"].min())
        winners = group[
            np.isclose(group["origin_mae"], best_value)
        ]
        winner_labels = sorted(winners["method"].tolist())
        chosen = winner_labels[0]

        router_origin = router[
            router["outer_origin"] == origin
        ]
        router_mae = float(
            router_origin["abs_error"].mean()
        )
        selected = sorted(
            router_origin["selected_expert"]
            .dropna()
            .unique()
            .tolist()
        )
        selected_label = (
            selected[0]
            if len(selected) == 1
            else "|".join(selected)
        )

        origin_rows.append({
            "outer_origin": int(origin),
            "router_method": router_method,
            "selected_expert": selected_label,
            "oracle_experts": "|".join(winner_labels),
            "router_origin_mae": router_mae,
            "oracle_origin_mae": best_value,
            "origin_regret": router_mae - best_value,
            "selection_correct": selected_label in winner_labels,
            "post_hoc": True,
            "note": POST_HOC_NOTE,
        })

        chosen_points = fixed[
            (fixed["outer_origin"] == origin)
            & (fixed["method"] == chosen)
        ]
        for _, point in chosen_points.iterrows():
            oracle_point_rows.append({
                "outer_origin": int(origin),
                "horizon_step": int(point["horizon_step"]),
                "oracle_expert": chosen,
                "abs_error": float(point["abs_error"]),
                "post_hoc": True,
            })

    by_origin = pd.DataFrame(origin_rows)
    oracle_points = pd.DataFrame(oracle_point_rows)

    horizon_rows = []
    for (origin, step), group in fixed.groupby(
        ["outer_origin", "horizon_step"]
    ):
        best_value = float(group["abs_error"].min())
        winners = group[
            np.isclose(group["abs_error"], best_value)
        ]
        horizon_rows.append({
            "outer_origin": int(origin),
            "horizon_step": int(step),
            "oracle_abs_error": best_value,
            "oracle_experts": "|".join(
                sorted(winners["method"].tolist())
            ),
            "post_hoc": True,
            "note": (
                "Horizon-specific ex-post upper bound. "
                "Not a routing rule."
            ),
        })

    horizon_oracle = pd.DataFrame(horizon_rows)

    summary = pd.DataFrame([{
        "router_method": router_method,
        "router_mae": float(router["abs_error"].mean()),
        "oracle_mae": float(
            oracle_points["abs_error"].mean()
        ),
        "router_regret": float(
            router["abs_error"].mean()
            - oracle_points["abs_error"].mean()
        ),
        "oracle_gap": float(
            by_origin["origin_regret"].mean()
        ),
        "selection_accuracy": float(
            by_origin["selection_correct"].mean()
        ),
        "n_outer_origins": int(
            by_origin["outer_origin"].nunique()
        ),
        "horizon_oracle_mae": float(
            horizon_oracle["oracle_abs_error"].mean()
        ),
        "post_hoc": True,
        "note": POST_HOC_NOTE,
    }])

    return {
        "summary": summary,
        "by_origin": by_origin,
        "horizon_oracle": horizon_oracle,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc oracle diagnostics for a saved "
            "nested evaluation. This does not reroute."
        )
    )
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument(
        "--router",
        type=str,
        default="dynamic_hard_router",
    )
    args = parser.parse_args(argv)

    directory = Path(args.results)
    predictions = pd.read_csv(
        directory / "predictions.csv"
    )
    tables = oracle_analysis(
        predictions,
        router_method=args.router,
    )

    for name, frame in tables.items():
        frame.to_csv(
            directory / f"oracle_{name}.csv",
            index=False,
        )

    print(tables["summary"].to_string(index=False))
    print(POST_HOC_NOTE)
    print(f"Saved under: {directory}")


if __name__ == "__main__":
    main()
