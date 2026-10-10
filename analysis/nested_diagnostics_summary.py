import os

import numpy as np
import pandas as pd


RESULTS_DIR = "results"


def load_results(
    results_dir=RESULTS_DIR,
):

    diagnostics_path = os.path.join(
        results_dir,
        "nested_routing_diagnostics.csv",
    )

    predictions_path = os.path.join(
        results_dir,
        "nested_routing_predictions.csv",
    )

    experts_path = os.path.join(
        results_dir,
        "nested_routing_selected_experts.csv",
    )


    diagnostics = pd.read_csv(
        diagnostics_path
    )

    predictions = pd.read_csv(
        predictions_path
    )

    experts = pd.read_csv(
        experts_path
    )


    return (
        diagnostics,
        predictions,
        experts,
    )


def build_origin_horizon_summary(
    diagnostics,
    predictions,
    experts,
):

    rows = []


    origin_horizon_pairs = (

        diagnostics[
            [
                "outer_origin",
                "horizon",
            ]
        ]

        .drop_duplicates()

        .sort_values(
            [
                "outer_origin",
                "horizon",
            ]
        )
    )


    for _, pair in (
        origin_horizon_pairs.iterrows()
    ):

        outer_origin = int(
            pair[
                "outer_origin"
            ]
        )

        horizon = int(
            pair[
                "horizon"
            ]
        )


        # =========================================
        # Expert-level rows
        # =========================================

        subset = diagnostics[
            (
                diagnostics[
                    "outer_origin"
                ]
                == outer_origin
            )
            &
            (
                diagnostics[
                    "horizon"
                ]
                == horizon
            )
        ].copy()


        # =========================================
        # Hard-routing expert
        #
        # Rank 1 expert from overall routing.
        # =========================================

        expert_subset = experts[
            experts[
                "outer_origin"
            ]
            == outer_origin
        ]


        hard_row = (
            expert_subset[
                expert_subset[
                    "rank"
                ]
                == 1
            ]
        )


        hard_model = (
            hard_row.iloc[
                0
            ][
                "model"
            ]
        )

        hard_window = (
            hard_row.iloc[
                0
            ][
                "window_type"
            ]
        )


        # =========================================
        # State router top-weight expert
        # =========================================

        state_top_idx = (
            subset[
                "state_horizon_weight"
            ]
            .idxmax()
        )


        state_top_row = (
            subset.loc[
                state_top_idx
            ]
        )


        state_top_model = (
            state_top_row[
                "model"
            ]
        )

        state_top_window = (
            state_top_row[
                "window_type"
            ]
        )


        # =========================================
        # Horizon-router top expert
        # =========================================

        horizon_top_idx = (
            subset[
                "horizon_weight"
            ]
            .idxmax()
        )


        horizon_top_row = (
            subset.loc[
                horizon_top_idx
            ]
        )


        horizon_top_model = (
            horizon_top_row[
                "model"
            ]
        )

        horizon_top_window = (
            horizon_top_row[
                "window_type"
            ]
        )


        # =========================================
        # Strategy predictions
        # =========================================

        pred_subset = predictions[
            (
                predictions[
                    "outer_origin"
                ]
                == outer_origin
            )
            &
            (
                predictions[
                    "horizon"
                ]
                == horizon
            )
        ]


        def get_strategy_row(
            strategy,
        ):

            match = pred_subset[
                pred_subset[
                    "strategy"
                ]
                == strategy
            ]

            if len(match) == 0:

                return None

            return match.iloc[
                0
            ]


        hard_pred = get_strategy_row(
            "hard_routing_single"
        )

        state_pred = get_strategy_row(
            "state_horizon_weight"
        )

        horizon_pred = get_strategy_row(
            "horizon_weight"
        )

        overall_pred = get_strategy_row(
            "overall_weight"
        )

        equal_pred = get_strategy_row(
            "equal_weight"
        )


        actual = float(
            hard_pred[
                "actual"
            ]
        )


        hard_abs_error = float(
            hard_pred[
                "abs_error"
            ]
        )

        state_abs_error = float(
            state_pred[
                "abs_error"
            ]
        )

        horizon_abs_error = float(
            horizon_pred[
                "abs_error"
            ]
        )

        overall_abs_error = float(
            overall_pred[
                "abs_error"
            ]
        )

        equal_abs_error = float(
            equal_pred[
                "abs_error"
            ]
        )


        # =========================================
        # Did state routing help?
        #
        # Positive:
        # state router is better than hard routing.
        # =========================================

        state_improvement = (
            hard_abs_error
            - state_abs_error
        )


        horizon_improvement = (
            hard_abs_error
            - horizon_abs_error
        )


        # =========================================
        # Agreement indicators
        # =========================================

        state_agrees_hard = bool(
            (
                state_top_model
                == hard_model
            )
            and
            (
                state_top_window
                == hard_window
            )
        )


        horizon_agrees_hard = bool(
            (
                horizon_top_model
                == hard_model
            )
            and
            (
                horizon_top_window
                == hard_window
            )
        )


        # =========================================
        # Router diagnostics
        # =========================================

        state_ess = float(
            subset[
                "selected_pool_state_ess"
            ]
            .iloc[
                0
            ]
        )

        state_entropy = float(
            subset[
                "selected_pool_state_entropy"
            ]
            .iloc[
                0
            ]
        )

        state_max_weight = float(
            subset[
                "state_horizon_weight"
            ]
            .max()
        )


        # Historical localization ESS
        #
        # This is different from selected-pool ESS.
        # It measures how many historical states
        # effectively support localized loss.
        # =========================================

        historical_ess = np.nan

        if (
            "historical_state_ess"
            in subset.columns
        ):

            valid_historical_ess = (
                subset[
                    "historical_state_ess"
                ]
                .dropna()
            )

            if len(
                valid_historical_ess
            ) > 0:

                historical_ess = float(
                    valid_historical_ess.mean()
                )


        rows.append({

            "outer_origin":
                outer_origin,

            "horizon":
                horizon,

            "actual":
                actual,

            "hard_model":
                hard_model,

            "hard_window":
                hard_window,

            "state_top_model":
                state_top_model,

            "state_top_window":
                state_top_window,

            "horizon_top_model":
                horizon_top_model,

            "horizon_top_window":
                horizon_top_window,

            "state_agrees_hard":
                state_agrees_hard,

            "horizon_agrees_hard":
                horizon_agrees_hard,

            "hard_abs_error":
                hard_abs_error,

            "state_abs_error":
                state_abs_error,

            "horizon_abs_error":
                horizon_abs_error,

            "overall_abs_error":
                overall_abs_error,

            "equal_abs_error":
                equal_abs_error,

            "state_improvement_vs_hard":
                state_improvement,

            "horizon_improvement_vs_hard":
                horizon_improvement,

            "state_pool_ess":
                state_ess,

            "historical_state_ess":
                historical_ess,

            "state_entropy":
                state_entropy,

            "state_max_weight":
                state_max_weight,
        })


    return pd.DataFrame(
        rows
    )


def print_summary(
    summary,
):

    print(
        "\n==================================="
    )

    print(
        "Origin × Horizon Router Diagnostics"
    )

    print(
        "==================================="
    )


    display_columns = [

        "outer_origin",
        "horizon",

        "hard_model",
        "state_top_model",

        "state_agrees_hard",

        "hard_abs_error",
        "state_abs_error",

        "state_improvement_vs_hard",

        "state_pool_ess",
        "historical_state_ess",

        "state_max_weight",
    ]


    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )


    # =========================================
    # Agreement
    # =========================================

    agreement_rate = float(
        summary[
            "state_agrees_hard"
        ].mean()
    )


    horizon_agreement_rate = float(
        summary[
            "horizon_agrees_hard"
        ].mean()
    )


    print(
        "\n==================================="
    )

    print(
        "Router Agreement"
    )

    print(
        "==================================="
    )


    print(
        "State router agrees with hard router:"
    )

    print(
        f"{agreement_rate:.3f}"
    )


    print(
        "\nHorizon router agrees with hard router:"
    )

    print(
        f"{horizon_agreement_rate:.3f}"
    )


    # =========================================
    # When state changes winner
    # =========================================

    changed = summary[
        ~summary[
            "state_agrees_hard"
        ]
    ]


    unchanged = summary[
        summary[
            "state_agrees_hard"
        ]
    ]


    print(
        "\n==================================="
    )

    print(
        "Effect of State Router"
    )

    print(
        "==================================="
    )


    if len(changed) > 0:

        print(
            "\nWhen state router DISAGREES "
            "with hard routing:"
        )

        print(
            f"n = {len(changed)}"
        )

        print(
            "Mean hard MAE:"
        )

        print(
            f"{changed['hard_abs_error'].mean():.3f}"
        )

        print(
            "Mean state-router MAE:"
        )

        print(
            f"{changed['state_abs_error'].mean():.3f}"
        )

        print(
            "Mean improvement "
            "(positive = state better):"
        )

        print(
            f"{changed['state_improvement_vs_hard'].mean():.3f}"
        )

        win_rate = float(
            (
                changed[
                    "state_improvement_vs_hard"
                ]
                > 0
            )
            .mean()
        )

        print(
            "State-router win rate "
            "when disagreeing:"
        )

        print(
            f"{win_rate:.3f}"
        )


    if len(unchanged) > 0:

        print(
            "\nWhen state router AGREES "
            "with hard routing:"
        )

        print(
            f"n = {len(unchanged)}"
        )

        print(
            "Mean hard MAE:"
        )

        print(
            f"{unchanged['hard_abs_error'].mean():.3f}"
        )

        print(
            "Mean state-router MAE:"
        )

        print(
            f"{unchanged['state_abs_error'].mean():.3f}"
        )


    # =========================================
    # By horizon
    # =========================================

    print(
        "\n==================================="
    )

    print(
        "State Router by Horizon"
    )

    print(
        "==================================="
    )


    horizon_summary = (

        summary

        .groupby(
            "horizon"
        )

        .agg(

            agreement_rate=(
                "state_agrees_hard",
                "mean",
            ),

            hard_mae=(
                "hard_abs_error",
                "mean",
            ),

            state_mae=(
                "state_abs_error",
                "mean",
            ),

            mean_improvement=(
                "state_improvement_vs_hard",
                "mean",
            ),

            mean_pool_ess=(
                "state_pool_ess",
                "mean",
            ),

            mean_historical_ess=(
                "historical_state_ess",
                "mean",
            ),

            mean_max_weight=(
                "state_max_weight",
                "mean",
            ),
        )

        .reset_index()
    )


    print(
        horizon_summary.to_string(
            index=False
        )
    )


    return horizon_summary


if __name__ == "__main__":

    (
        diagnostics,
        predictions,
        experts,
    ) = load_results()


    summary = (
        build_origin_horizon_summary(

            diagnostics=
                diagnostics,

            predictions=
                predictions,

            experts=
                experts,
        )
    )


    horizon_summary = (
        print_summary(
            summary
        )
    )


    output_path = os.path.join(
        RESULTS_DIR,
        "nested_router_diagnostic_summary.csv",
    )


    horizon_output_path = os.path.join(
        RESULTS_DIR,
        "nested_router_horizon_diagnostic_summary.csv",
    )


    summary.to_csv(
        output_path,
        index=False,
    )


    horizon_summary.to_csv(
        horizon_output_path,
        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        horizon_output_path
    )