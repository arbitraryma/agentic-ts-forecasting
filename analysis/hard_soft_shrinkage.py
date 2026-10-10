# LEGACY / DEBUG SCRIPT.
# Reads older saved result files.
# Not the paper benchmark or the paper oracle diagnostic.
# Importing this file does not launch the analysis.

import os

import numpy as np
import pandas as pd




def main():
    RESULTS_DIR = "results"

    PREDICTIONS_PATH = os.path.join(
        RESULTS_DIR,
        "nested_routing_predictions.csv",
    )

    OUTPUT_PATH = os.path.join(
        RESULTS_DIR,
        "hard_soft_shrinkage_diagnostic.csv",
    )

    HORIZON_OUTPUT_PATH = os.path.join(
        RESULTS_DIR,
        "hard_soft_shrinkage_horizon_diagnostic.csv",
    )


    # =========================================================
    # Settings
    # =========================================================

    LAMBDAS = np.round(
        np.arange(
            0.0,
            1.01,
            0.1,
        ),
        2,
    )


    # =========================================================
    # Load predictions
    # =========================================================

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )


    required_strategies = {
        "hard_routing_single",
        "state_horizon_weight",
    }


    available_strategies = set(
        predictions[
            "strategy"
        ].unique()
    )


    missing = (
        required_strategies
        - available_strategies
    )


    if missing:

        raise ValueError(
            "Missing required strategies: "
            f"{sorted(missing)}"
        )


    # =========================================================
    # Reshape hard and soft predictions
    # =========================================================

    hard = (
        predictions[
            predictions[
                "strategy"
            ]
            == "hard_routing_single"
        ][
            [
                "outer_origin",
                "horizon",
                "actual",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction":
                    "hard_prediction"
            }
        )
    )


    state = (
        predictions[
            predictions[
                "strategy"
            ]
            == "state_horizon_weight"
        ][
            [
                "outer_origin",
                "horizon",
                "prediction",
            ]
        ]
        .rename(
            columns={
                "prediction":
                    "state_prediction"
            }
        )
    )


    merged = hard.merge(
        state,
        on=[
            "outer_origin",
            "horizon",
        ],
        how="inner",
    )


    if len(merged) == 0:

        raise RuntimeError(
            "No matched hard/state predictions."
        )


    # =========================================================
    # Evaluate shrinkage
    # =========================================================

    overall_rows = []

    horizon_rows = []


    for lam in LAMBDAS:

        temp = merged.copy()


        # -----------------------------------------------------
        # Convex combination
        #
        # lambda = 0:
        # pure hard routing
        #
        # lambda = 1:
        # pure state-horizon soft routing
        # -----------------------------------------------------

        temp[
            "shrinkage_prediction"
        ] = (

            (1.0 - lam)
            * temp[
                "hard_prediction"
            ]

            +

            lam
            * temp[
                "state_prediction"
            ]
        )


        temp[
            "error"
        ] = (

            temp[
                "actual"
            ]

            -

            temp[
                "shrinkage_prediction"
            ]
        )


        temp[
            "abs_error"
        ] = np.abs(
            temp[
                "error"
            ]
        )


        temp[
            "sq_error"
        ] = (
            temp[
                "error"
            ]
            ** 2
        )


        # =====================================================
        # Overall
        # =====================================================

        overall_rows.append({

            "lambda":
                float(
                    lam
                ),

            "mae":
                float(
                    temp[
                        "abs_error"
                    ].mean()
                ),

            "rmse":
                float(
                    np.sqrt(
                        temp[
                            "sq_error"
                        ].mean()
                    )
                ),

            "n_forecasts":
                int(
                    len(
                        temp
                    )
                ),
        })


        # =====================================================
        # Horizon-specific
        # =====================================================

        for (
            horizon,
            group,
        ) in temp.groupby(
            "horizon"
        ):

            horizon_rows.append({

                "lambda":
                    float(
                        lam
                    ),

                "horizon":
                    int(
                        horizon
                    ),

                "mae":
                    float(
                        group[
                            "abs_error"
                        ].mean()
                    ),

                "rmse":
                    float(
                        np.sqrt(
                            group[
                                "sq_error"
                            ].mean()
                        )
                    ),

                "n_forecasts":
                    int(
                        len(
                            group
                        )
                    ),
            })


    # =========================================================
    # Tables
    # =========================================================

    overall_summary = pd.DataFrame(
        overall_rows
    )


    horizon_summary = pd.DataFrame(
        horizon_rows
    )


    # =========================================================
    # Best diagnostic lambdas
    # =========================================================

    best_overall = (
        overall_summary

        .sort_values(
            "mae"
        )

        .iloc[
            0
        ]
    )


    best_by_horizon = (

        horizon_summary

        .sort_values(
            [
                "horizon",
                "mae",
            ]
        )

        .groupby(
            "horizon",
            as_index=False,
        )

        .first()
    )


    # =========================================================
    # Print
    # =========================================================

    print(
        "\n==================================="
    )

    print(
        "Hard → State Shrinkage Diagnostic"
    )

    print(
        "==================================="
    )


    print(
        overall_summary.to_string(
            index=False
        )
    )


    print(
        "\n==================================="
    )

    print(
        "Best Overall Lambda"
    )

    print(
        "==================================="
    )


    print(
        best_overall.to_string()
    )


    print(
        "\n==================================="
    )

    print(
        "Best Lambda by Horizon"
    )

    print(
        "==================================="
    )


    print(
        best_by_horizon.to_string(
            index=False
        )
    )


    # =========================================================
    # Improvement relative to hard routing
    # =========================================================

    hard_row = (
        overall_summary[
            overall_summary[
                "lambda"
            ]
            == 0.0
        ]
        .iloc[
            0
        ]
    )


    hard_mae = float(
        hard_row[
            "mae"
        ]
    )


    best_mae = float(
        best_overall[
            "mae"
        ]
    )


    improvement = (
        hard_mae
        - best_mae
    )


    relative_improvement = (

        improvement
        /
        hard_mae

        if hard_mae > 0

        else np.nan
    )


    print(
        "\n==================================="
    )

    print(
        "Diagnostic Improvement"
    )

    print(
        "==================================="
    )


    print(
        f"Hard-routing MAE: "
        f"{hard_mae:.6f}"
    )


    print(
        f"Best shrinkage MAE: "
        f"{best_mae:.6f}"
    )


    print(
        f"Absolute improvement: "
        f"{improvement:.6f}"
    )


    print(
        f"Relative improvement: "
        f"{100 * relative_improvement:.2f}%"
    )


    # =========================================================
    # Save
    # =========================================================

    overall_summary.to_csv(
        OUTPUT_PATH,
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
        OUTPUT_PATH
    )

    print(
        HORIZON_OUTPUT_PATH
    )


    # =========================================================
    # Important warning
    # =========================================================

    print(
        "\nNOTE:"
    )

    print(
        "This is a post-hoc diagnostic using outer-test "
        "errors. The best lambda found here must NOT be "
        "reported as an unbiased forecasting result."
    )

    print(
        "If non-zero shrinkage looks promising, lambda "
        "must later be selected using training/validation "
        "data only inside the nested evaluation."
    )


if __name__ == "__main__":
    main()
