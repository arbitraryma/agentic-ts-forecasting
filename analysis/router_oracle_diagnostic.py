# LEGACY / DEBUG SCRIPT.
# Reads older saved result files.
# The paper oracle diagnostic is analysis/oracle_analysis.py.
# Importing this file does not launch the analysis.

import os

import numpy as np
import pandas as pd




def main():
    RESULTS_DIR = "results"


    FIXED_PREDICTIONS_PATH = os.path.join(
        RESULTS_DIR,
        "nested_fixed_expert_predictions.csv",
    )

    ROUTER_PREDICTIONS_PATH = os.path.join(
        RESULTS_DIR,
        "nested_routing_predictions.csv",
    )

    SELECTED_EXPERTS_PATH = os.path.join(
        RESULTS_DIR,
        "nested_routing_selected_experts.csv",
    )


    ORIGIN_OUTPUT_PATH = os.path.join(
        RESULTS_DIR,
        "router_oracle_by_origin.csv",
    )

    SUMMARY_OUTPUT_PATH = os.path.join(
        RESULTS_DIR,
        "router_oracle_summary.csv",
    )

    HORIZON_OUTPUT_PATH = os.path.join(
        RESULTS_DIR,
        "router_oracle_by_horizon.csv",
    )


    # =========================================================
    # Load data
    # =========================================================

    fixed_predictions = pd.read_csv(
        FIXED_PREDICTIONS_PATH
    )

    router_predictions = pd.read_csv(
        ROUTER_PREDICTIONS_PATH
    )

    selected_experts = pd.read_csv(
        SELECTED_EXPERTS_PATH
    )


    # =========================================================
    # Basic checks
    # =========================================================

    required_fixed_columns = {
        "outer_origin",
        "horizon",
        "expert",
        "abs_error",
    }

    missing_fixed = (
        required_fixed_columns
        - set(
            fixed_predictions.columns
        )
    )

    if missing_fixed:

        raise ValueError(
            "Missing fixed-expert columns: "
            f"{sorted(missing_fixed)}"
        )


    required_router_columns = {
        "outer_origin",
        "horizon",
        "strategy",
        "abs_error",
    }

    missing_router = (
        required_router_columns
        - set(
            router_predictions.columns
        )
    )

    if missing_router:

        raise ValueError(
            "Missing router columns: "
            f"{sorted(missing_router)}"
        )


    # =========================================================
    # Router-selected expert per origin
    #
    # Rank 1 from the inner validation router.
    # =========================================================

    router_selected = (

        selected_experts[
            selected_experts[
                "rank"
            ]
            == 1
        ]
        .copy()
    )


    router_selected[
        "router_expert"
    ] = (

        router_selected[
            "model"
        ]
        +
        "__"
        +
        router_selected[
            "window_type"
        ]
    )


    router_selected = (
        router_selected[
            [
                "outer_origin",
                "router_expert",
            ]
        ]
    )


    # =========================================================
    # Outer-origin expert performance
    #
    # Aggregate across all horizons.
    # =========================================================

    expert_origin_perf = (

        fixed_predictions

        .groupby(
            [
                "outer_origin",
                "expert",
            ],
            as_index=False,
        )

        .agg(
            mae=(
                "abs_error",
                "mean",
            ),

            rmse=(
                "sq_error",
                lambda x:
                    float(
                        np.sqrt(
                            np.mean(
                                x
                            )
                        )
                    ),
            ),
        )
    )


    # =========================================================
    # Oracle winner per origin
    # =========================================================

    oracle_rows = []


    for (
        outer_origin,
        group,
    ) in expert_origin_perf.groupby(
        "outer_origin"
    ):

        group = (
            group

            .sort_values(
                "mae",
                ascending=True,
            )

            .reset_index(
                drop=True
            )
        )


        best = group.iloc[
            0
        ]


        second_best_mae = np.nan

        if len(
            group
        ) > 1:

            second_best_mae = float(
                group.iloc[
                    1
                ][
                    "mae"
                ]
            )


        oracle_rows.append({

            "outer_origin":
                int(
                    outer_origin
                ),

            "oracle_expert":
                best[
                    "expert"
                ],

            "oracle_mae":
                float(
                    best[
                        "mae"
                    ]
                ),

            "oracle_rmse":
                float(
                    best[
                        "rmse"
                    ]
                ),

            "second_best_mae":
                second_best_mae,

            "oracle_margin":
                (
                    second_best_mae
                    -
                    float(
                        best[
                            "mae"
                        ]
                    )
                    if pd.notna(
                        second_best_mae
                    )
                    else np.nan
                ),
        })


    oracle_by_origin = pd.DataFrame(
        oracle_rows
    )


    # =========================================================
    # Dynamic router performance per origin
    # =========================================================

    dynamic_router = (

        router_predictions[
            router_predictions[
                "strategy"
            ]
            == "dynamic_hard_router"
        ]
        .copy()
    )


    router_origin_perf = (

        dynamic_router

        .groupby(
            "outer_origin",
            as_index=False,
        )

        .agg(
            router_mae=(
                "abs_error",
                "mean",
            ),

            router_rmse=(
                "sq_error",
                lambda x:
                    float(
                        np.sqrt(
                            np.mean(
                                x
                            )
                        )
                    ),
            ),
        )
    )


    # =========================================================
    # Merge origin diagnostics
    # =========================================================

    origin_summary = (

        oracle_by_origin

        .merge(
            router_origin_perf,
            on="outer_origin",
            how="inner",
        )

        .merge(
            router_selected,
            on="outer_origin",
            how="left",
        )
    )


    origin_summary[
        "router_matches_oracle"
    ] = (

        origin_summary[
            "router_expert"
        ]
        ==
        origin_summary[
            "oracle_expert"
        ]
    )


    origin_summary[
        "router_regret"
    ] = (

        origin_summary[
            "router_mae"
        ]
        -
        origin_summary[
            "oracle_mae"
        ]
    )


    origin_summary[
        "router_regret_pct"
    ] = (

        origin_summary[
            "router_regret"
        ]
        /
        origin_summary[
            "oracle_mae"
        ]
        *
        100.0
    )


    # =========================================================
    # Horizon-level oracle
    #
    # Here the oracle can pick a different expert
    # separately for each horizon.
    # This is a stronger post-hoc upper bound.
    # =========================================================

    horizon_oracle_rows = []


    for (
        outer_origin,
        horizon,
    ), group in fixed_predictions.groupby(
        [
            "outer_origin",
            "horizon",
        ]
    ):

        group = (
            group

            .sort_values(
                "abs_error",
                ascending=True,
            )

            .reset_index(
                drop=True
            )
        )


        best = group.iloc[
            0
        ]


        router_match = (
            dynamic_router[
                (
                    dynamic_router[
                        "outer_origin"
                    ]
                    == outer_origin
                )
                &
                (
                    dynamic_router[
                        "horizon"
                    ]
                    == horizon
                )
            ]
        )


        if len(
            router_match
        ) == 0:

            continue


        router_row = (
            router_match.iloc[
                0
            ]
        )


        horizon_oracle_rows.append({

            "outer_origin":
                int(
                    outer_origin
                ),

            "horizon":
                int(
                    horizon
                ),

            "oracle_expert":
                best[
                    "expert"
                ],

            "oracle_abs_error":
                float(
                    best[
                        "abs_error"
                    ]
                ),

            "router_abs_error":
                float(
                    router_row[
                        "abs_error"
                    ]
                ),

            "router_regret":
                float(
                    router_row[
                        "abs_error"
                    ]
                    -
                    best[
                        "abs_error"
                    ]
                ),
        })


    horizon_oracle = pd.DataFrame(
        horizon_oracle_rows
    )


    # =========================================================
    # Global summary
    # =========================================================

    selection_accuracy = float(
        origin_summary[
            "router_matches_oracle"
        ]
        .mean()
    )


    mean_router_mae = float(
        origin_summary[
            "router_mae"
        ]
        .mean()
    )


    mean_oracle_mae = float(
        origin_summary[
            "oracle_mae"
        ]
        .mean()
    )


    mean_regret = float(
        origin_summary[
            "router_regret"
        ]
        .mean()
    )


    median_regret = float(
        origin_summary[
            "router_regret"
        ]
        .median()
    )


    oracle_gap_pct = (

        (
            mean_router_mae
            -
            mean_oracle_mae
        )
        /
        mean_oracle_mae
        *
        100.0
    )


    summary = pd.DataFrame(
        [
            {
                "n_outer_origins":
                    int(
                        len(
                            origin_summary
                        )
                    ),

                "selection_accuracy":
                    selection_accuracy,

                "mean_router_mae":
                    mean_router_mae,

                "mean_oracle_mae":
                    mean_oracle_mae,

                "mean_router_regret":
                    mean_regret,

                "median_router_regret":
                    median_regret,

                "oracle_gap_pct":
                    oracle_gap_pct,
            }
        ]
    )


    # =========================================================
    # Horizon summary
    # =========================================================

    horizon_summary = (

        horizon_oracle

        .groupby(
            "horizon",
            as_index=False,
        )

        .agg(
            router_mae=(
                "router_abs_error",
                "mean",
            ),

            oracle_mae=(
                "oracle_abs_error",
                "mean",
            ),

            mean_regret=(
                "router_regret",
                "mean",
            ),
        )
    )


    horizon_summary[
        "oracle_gap_pct"
    ] = (

        (
            horizon_summary[
                "router_mae"
            ]
            -
            horizon_summary[
                "oracle_mae"
            ]
        )
        /
        horizon_summary[
            "oracle_mae"
        ]
        *
        100.0
    )


    # =========================================================
    # Print
    # =========================================================

    print(
        "\n==================================="
    )

    print(
        "Dynamic Router vs Oracle"
    )

    print(
        "==================================="
    )


    display_columns = [

        "outer_origin",

        "router_expert",
        "oracle_expert",

        "router_matches_oracle",

        "router_mae",
        "oracle_mae",

        "router_regret",

        "oracle_margin",
    ]


    print(
        origin_summary[
            display_columns
        ].to_string(
            index=False
        )
    )


    print(
        "\n==================================="
    )

    print(
        "Overall Oracle Diagnostic"
    )

    print(
        "==================================="
    )


    print(
        summary.to_string(
            index=False
        )
    )


    print(
        "\n==================================="
    )

    print(
        "Horizon-Level Oracle Diagnostic"
    )

    print(
        "==================================="
    )


    print(
        horizon_summary.to_string(
            index=False
        )
    )


    # =========================================================
    # Extra interpretation aids
    # =========================================================

    n_correct = int(
        origin_summary[
            "router_matches_oracle"
        ]
        .sum()
    )


    n_total = int(
        len(
            origin_summary
        )
    )


    print(
        "\n==================================="
    )

    print(
        "Quick Interpretation"
    )

    print(
        "==================================="
    )


    print(
        f"Router selected the ex-post best "
        f"expert in {n_correct}/{n_total} "
        f"outer origins."
    )


    print(
        f"Selection accuracy: "
        f"{100 * selection_accuracy:.1f}%"
    )


    print(
        f"Mean router MAE: "
        f"{mean_router_mae:.3f}"
    )


    print(
        f"Mean oracle MAE: "
        f"{mean_oracle_mae:.3f}"
    )


    print(
        f"Mean router regret: "
        f"{mean_regret:.3f}"
    )


    print(
        f"Relative gap to oracle: "
        f"{oracle_gap_pct:.1f}%"
    )


    # =========================================================
    # Save
    # =========================================================

    origin_summary.to_csv(
        ORIGIN_OUTPUT_PATH,
        index=False,
    )


    summary.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False,
    )


    horizon_summary.to_csv(
        HORIZON_OUTPUT_PATH,
        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        ORIGIN_OUTPUT_PATH
    )

    print(
        SUMMARY_OUTPUT_PATH
    )

    print(
        HORIZON_OUTPUT_PATH
    )


    # =========================================================
    # Warning
    # =========================================================

    print(
        "\nNOTE:"
    )

    print(
        "The oracle uses outer-test outcomes and is "
        "therefore only a post-hoc diagnostic upper bound."
    )

    print(
        "It must never be used for model selection in the "
        "actual forecasting pipeline."
    )


if __name__ == "__main__":
    main()
