# LEGACY / DEBUG SCRIPT.
# Not the paper benchmark. Use:
#     python -m experiments.run_benchmark --config configs/airpassengers.yaml
# This file hard-codes an AirPassengers path.
# Importing it does not launch the experiment.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

import numpy as np
import pandas as pd

from diagnostics.basic import basic_diagnostics
from diagnostics.state import extract_state_features

from evaluation.rolling_cv import rolling_backtest

from models.registry import (
    MODEL_REGISTRY,
    get_model_class,
)


def main():

    from models.config_builder import build_model_kwargs

    from routing.candidate_router import route_candidates

    from routing.validation_router import compute_routing_scores

    from routing.horizon_router import (
        build_horizon_routing_table,
    )

    from routing.state_horizon_router import (
        build_state_horizon_routing_table,
    )


    # =========================================================
    # CLI
    # =========================================================

    parser = argparse.ArgumentParser(
        description=(
            "Nested rolling OOS evaluation with "
            "origin-level and horizon-specific hard routing."
        )
    )

    parser.add_argument(
        "--full",
        action="store_true",
    )

    args = parser.parse_args()


    # =========================================================
    # Dataset
    # =========================================================

    DATA_PATH = "data/AirPassengers.csv"
    TARGET_COLUMN = "#Passengers"


    # =========================================================
    # Router settings
    # =========================================================

    COMPUTE_WEIGHT = 0.10
    TEMPERATURE = 1.0

    STATE_BANDWIDTH = 2.0
    RECENCY_DECAY = 0.1

    TOP_K = 3

    COMPUTE_BUDGET = "medium"


    # =========================================================
    # Experiment settings
    # =========================================================

    if args.full:

        HORIZON = 3
        OUTER_WINDOWS = 5
        INNER_WINDOWS = 5

        TRAIN_WINDOW = 60

        WINDOW_TYPES = [
            "expanding",
            "sliding",
        ]

        FAST_MODEL_CONFIG = False

        MODEL_ALLOWLIST = None

    else:

        HORIZON = 3
        OUTER_WINDOWS = 3
        INNER_WINDOWS = 3

        TRAIN_WINDOW = 60

        WINDOW_TYPES = [
            "expanding",
        ]

        FAST_MODEL_CONFIG = True

        MODEL_ALLOWLIST = {
            "seasonal_naive",
            "auto_arima",
            "ets",
            "xgboost",
            "chronos2",
            "timesfm3",
        }


    # =========================================================
    # Helpers
    # =========================================================

    def normalize_weights(weights):

        weights = np.asarray(
            weights,
            dtype=float,
        )

        total = float(
            weights.sum()
        )

        if total <= 0:

            return (
                np.ones(
                    len(weights),
                    dtype=float,
                )
                / len(weights)
            )

        return weights / total


    def find_routing_row(
        routing_table,
        model_name,
        window_type,
        horizon=None,
    ):

        mask = (
            (
                routing_table["model"]
                == model_name
            )
            &
            (
                routing_table["window_type"]
                == window_type
            )
        )

        if (
            horizon is not None
            and
            "horizon"
            in routing_table.columns
        ):

            mask = (
                mask
                &
                (
                    routing_table["horizon"]
                    == horizon
                )
            )

        matches = routing_table[
            mask
        ]

        if len(matches) == 0:

            return None

        return matches.iloc[0]


    def get_table_weights(
        routing_table,
        models,
        window_types,
        horizon=None,
    ):

        weights = []

        for model_name, window_type in zip(
            models,
            window_types,
        ):

            row = find_routing_row(
                routing_table,
                model_name,
                window_type,
                horizon,
            )

            if row is None:

                weights.append(
                    0.0
                )

            else:

                weights.append(
                    float(
                        row[
                            "routing_weight"
                        ]
                    )
                )

        return normalize_weights(
            weights
        )


    def fit_one_expert(
        model_name,
        window_type,
        train_y,
        diagnostics,
        horizon,
    ):

        model_class = get_model_class(
            model_name
        )

        model_kwargs = build_model_kwargs(
            model_name=model_name,
            diagnostics=diagnostics,
            fast=FAST_MODEL_CONFIG,
            horizon=horizon,
            n_series=1,
        )

        if window_type == "sliding":

            expert_train = (
                train_y.iloc[
                    -TRAIN_WINDOW:
                ]
                .reset_index(
                    drop=True
                )
            )

        else:

            expert_train = (
                train_y
                .copy()
                .reset_index(
                    drop=True
                )
            )

        model = model_class(
            **model_kwargs
        )

        model.fit(
            expert_train
        )

        forecast = model.predict(
            horizon=horizon
        )

        forecast = np.asarray(
            forecast,
            dtype=float,
        )

        forecast = np.squeeze(
            forecast
        )

        if forecast.ndim != 1:

            raise ValueError(
                f"{model_name} returned "
                f"shape {forecast.shape}"
            )

        if len(forecast) != horizon:

            raise ValueError(
                f"{model_name} returned "
                f"{len(forecast)} forecasts, "
                f"expected {horizon}"
            )

        return forecast


    def get_horizon_hard_winners(
        inner_results,
        horizon,
    ):
        """
        For each forecast horizon h,
        choose the expert with the lowest
        mean inner-CV absolute error.
        """

        winners = []

        for h in range(
            1,
            horizon + 1,
        ):

            loss_col = (
                f"abs_error_h{h}"
            )

            if loss_col not in inner_results.columns:

                raise ValueError(
                    f"Missing column: {loss_col}"
                )

            table = (
                inner_results

                .groupby(
                    [
                        "family",
                        "model",
                        "window_type",
                    ],
                    as_index=False,
                )[loss_col]

                .mean()

                .sort_values(
                    loss_col,
                    ascending=True,
                )

                .reset_index(
                    drop=True
                )
            )

            best = table.iloc[
                0
            ]

            winners.append({

                "horizon":
                    h,

                "family":
                    best[
                        "family"
                    ],

                "model":
                    best[
                        "model"
                    ],

                "window_type":
                    best[
                        "window_type"
                    ],

                "validation_loss":
                    float(
                        best[
                            loss_col
                        ]
                    ),
            })

        return pd.DataFrame(
            winners
        )


    # =========================================================
    # Load data
    # =========================================================

    df = pd.read_csv(
        DATA_PATH
    )

    y = (
        df[
            TARGET_COLUMN
        ]
        .astype(float)
        .reset_index(
            drop=True
        )
    )


    # =========================================================
    # Outer evaluation design
    # =========================================================

    total_outer_test_size = (
        OUTER_WINDOWS
        * HORIZON
    )

    initial_outer_train_size = (
        len(y)
        - total_outer_test_size
    )


    # =========================================================
    # Storage
    # =========================================================

    prediction_rows = []

    selected_expert_rows = []

    horizon_hard_selection_rows = []

    fixed_expert_prediction_rows = []


    # =========================================================
    # OUTER LOOP
    # =========================================================

    for outer_idx in range(
        OUTER_WINDOWS
    ):

        outer_train_end = (
            initial_outer_train_size
            +
            outer_idx
            * HORIZON
        )

        outer_test_start = (
            outer_train_end
        )

        outer_test_end = (
            outer_test_start
            + HORIZON
        )

        outer_train = (
            y.iloc[
                :outer_train_end
            ]
            .reset_index(
                drop=True
            )
        )

        outer_test = (
            y.iloc[
                outer_test_start:
                outer_test_end
            ]
            .reset_index(
                drop=True
            )
        )

        actual = (
            outer_test
            .to_numpy(
                dtype=float
            )
        )

        print(
            "\n==================================="
        )

        print(
            f"OUTER ORIGIN "
            f"{outer_idx + 1}"
        )

        print(
            "==================================="
        )


        # =====================================================
        # Diagnostics
        # =====================================================

        diagnostics = (
            basic_diagnostics(
                outer_train
            )
        )

        seasonal_period = (
            diagnostics.get(
                "seasonal_period"
            )
        )

        _ = extract_state_features(
            outer_train,
            seasonal_period=
                seasonal_period,
        )


        # =====================================================
        # Candidate routing
        # =====================================================

        candidates, _ = (
            route_candidates(

                diagnostics=
                    diagnostics,

                horizon=
                    HORIZON,

                n_series=
                    1,

                has_covariates=
                    False,

                compute_budget=
                    COMPUTE_BUDGET,
            )
        )


        if MODEL_ALLOWLIST is not None:

            candidates = [
                model_name
                for model_name
                in candidates
                if model_name
                in MODEL_ALLOWLIST
            ]


        # =====================================================
        # INNER CV
        # =====================================================

        inner_results = []


        for model_name in candidates:

            config = (
                MODEL_REGISTRY[
                    model_name
                ]
            )

            if not config[
                "enabled"
            ]:

                continue

            print(
                f"Inner CV: "
                f"{model_name}"
            )

            try:

                model_class = (
                    get_model_class(
                        model_name
                    )
                )

                model_kwargs = (
                    build_model_kwargs(

                        model_name=
                            model_name,

                        diagnostics=
                            diagnostics,

                        fast=
                            FAST_MODEL_CONFIG,

                        horizon=
                            HORIZON,

                        n_series=
                            1,
                    )
                )


                for window_type in WINDOW_TYPES:

                    kwargs = {

                        "model_class":
                            model_class,

                        "y":
                            outer_train,

                        "horizon":
                            HORIZON,

                        "n_windows":
                            INNER_WINDOWS,

                        "model_kwargs":
                            model_kwargs,

                        "window_type":
                            window_type,

                        "seasonal_period":
                            seasonal_period,
                    }

                    if window_type == "sliding":

                        kwargs[
                            "train_window"
                        ] = TRAIN_WINDOW

                    result = (
                        rolling_backtest(
                            **kwargs
                        )
                    )

                    result[
                        "model"
                    ] = model_name

                    result[
                        "family"
                    ] = config[
                        "family"
                    ]

                    inner_results.append(
                        result
                    )

            except Exception as e:

                print(
                    f"FAILED {model_name}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )


        inner_results = pd.concat(
            inner_results,
            ignore_index=True,
        )


        # =====================================================
        # Origin-level overall routing
        # =====================================================

        overall_summary = (
            inner_results

            .groupby(
                [
                    "family",
                    "model",
                    "window_type",
                ]
            )[
                [
                    "mae",
                    "rmse",
                    "runtime",
                ]
            ]

            .mean()

            .reset_index()
        )


        overall_routing = (
            compute_routing_scores(

                validation_summary=
                    overall_summary,

                metric=
                    "mae",

                compute_weight=
                    COMPUTE_WEIGHT,

                temperature=
                    TEMPERATURE,
            )
        )


        # =====================================================
        # Origin-level hard winner
        # =====================================================

        hard_winner = (
            overall_routing

            .sort_values(
                "utility",
                ascending=False,
            )

            .iloc[
                0
            ]
        )


        hard_model = (
            hard_winner[
                "model"
            ]
        )

        hard_window = (
            hard_winner[
                "window_type"
            ]
        )


        selected_expert_rows.append({

            "outer_origin":
                outer_idx + 1,

            "rank":
                1,

            "model":
                hard_model,

            "window_type":
                hard_window,
        })


        hard_forecast = (
            fit_one_expert(

                model_name=
                    hard_model,

                window_type=
                    hard_window,

                train_y=
                    outer_train,

                diagnostics=
                    diagnostics,

                horizon=
                    HORIZON,
            )
        )


        # =====================================================
        # Horizon-specific hard routing
        # =====================================================

        horizon_winners = (
            get_horizon_hard_winners(
                inner_results=
                    inner_results,
                horizon=
                    HORIZON,
            )
        )


        horizon_hard_forecast = np.zeros(
            HORIZON,
            dtype=float,
        )


        # Cache fitted expert forecasts so the same
        # expert does not need refitting repeatedly.
        forecast_cache = {}


        for _, winner in (
            horizon_winners.iterrows()
        ):

            h = int(
                winner[
                    "horizon"
                ]
            )

            model_name = (
                winner[
                    "model"
                ]
            )

            window_type = (
                winner[
                    "window_type"
                ]
            )

            cache_key = (
                model_name,
                window_type,
            )


            if cache_key not in forecast_cache:

                forecast_cache[
                    cache_key
                ] = fit_one_expert(

                    model_name=
                        model_name,

                    window_type=
                        window_type,

                    train_y=
                        outer_train,

                    diagnostics=
                        diagnostics,

                    horizon=
                        HORIZON,
                )


            forecast = (
                forecast_cache[
                    cache_key
                ]
            )


            horizon_hard_forecast[
                h - 1
            ] = forecast[
                h - 1
            ]


            horizon_hard_selection_rows.append({

                "outer_origin":
                    outer_idx + 1,

                "horizon":
                    h,

                "model":
                    model_name,

                "window_type":
                    window_type,

                "validation_loss":
                    winner[
                        "validation_loss"
                    ],
            })


        # =====================================================
        # Fixed expert OOS forecasts
        # =====================================================

        successful_experts = (
            overall_summary[
                [
                    "family",
                    "model",
                    "window_type",
                ]
            ]
            .drop_duplicates()
            .reset_index(
                drop=True
            )
        )


        for _, row in (
            successful_experts.iterrows()
        ):

            model_name = (
                row[
                    "model"
                ]
            )

            window_type = (
                row[
                    "window_type"
                ]
            )

            expert_label = (
                f"{model_name}__"
                f"{window_type}"
            )


            cache_key = (
                model_name,
                window_type,
            )


            try:

                if cache_key in forecast_cache:

                    forecast = (
                        forecast_cache[
                            cache_key
                        ]
                    )

                else:

                    forecast = fit_one_expert(

                        model_name=
                            model_name,

                        window_type=
                            window_type,

                        train_y=
                            outer_train,

                        diagnostics=
                            diagnostics,

                        horizon=
                            HORIZON,
                    )

                    forecast_cache[
                        cache_key
                    ] = forecast


                for h in range(
                    HORIZON
                ):

                    error = (
                        actual[
                            h
                        ]
                        -
                        forecast[
                            h
                        ]
                    )


                    fixed_expert_prediction_rows.append({

                        "outer_origin":
                            outer_idx + 1,

                        "horizon":
                            h + 1,

                        "model":
                            model_name,

                        "window_type":
                            window_type,

                        "expert":
                            expert_label,

                        "actual":
                            float(
                                actual[
                                    h
                                ]
                            ),

                        "prediction":
                            float(
                                forecast[
                                    h
                                ]
                            ),

                        "error":
                            float(
                                error
                            ),

                        "abs_error":
                            float(
                                abs(
                                    error
                                )
                            ),

                        "sq_error":
                            float(
                                error ** 2
                            ),
                    })


            except Exception as e:

                print(
                    f"Fixed expert failed "
                    f"{expert_label}: "
                    f"{e}"
                )


        # =====================================================
        # Store the two hard-routing strategies
        # =====================================================

        strategy_forecasts = {

            "dynamic_hard_router":
                hard_forecast,

            "dynamic_horizon_hard_router":
                horizon_hard_forecast,
        }


        for (
            strategy,
            forecast,
        ) in strategy_forecasts.items():

            for h in range(
                HORIZON
            ):

                error = (
                    actual[
                        h
                    ]
                    -
                    forecast[
                        h
                    ]
                )


                prediction_rows.append({

                    "outer_origin":
                        outer_idx + 1,

                    "horizon":
                        h + 1,

                    "strategy":
                        strategy,

                    "actual":
                        float(
                            actual[
                                h
                            ]
                        ),

                    "prediction":
                        float(
                            forecast[
                                h
                            ]
                        ),

                    "error":
                        float(
                            error
                        ),

                    "abs_error":
                        float(
                            abs(
                                error
                            )
                        ),

                    "sq_error":
                        float(
                            error ** 2
                        ),
                })


    # =========================================================
    # DataFrames
    # =========================================================

    predictions = pd.DataFrame(
        prediction_rows
    )

    fixed_predictions = pd.DataFrame(
        fixed_expert_prediction_rows
    )

    selected_experts = pd.DataFrame(
        selected_expert_rows
    )

    horizon_selections = pd.DataFrame(
        horizon_hard_selection_rows
    )


    # =========================================================
    # Strategy summary
    # =========================================================

    summary_rows = []


    for (
        strategy,
        group,
    ) in predictions.groupby(
        "strategy"
    ):

        summary_rows.append({

            "strategy":
                strategy,

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


    summary = (
        pd.DataFrame(
            summary_rows
        )

        .sort_values(
            "mae"
        )

        .reset_index(
            drop=True
        )
    )


    # =========================================================
    # Horizon summary
    # =========================================================

    horizon_rows = []


    for (
        strategy,
        horizon,
    ), group in predictions.groupby(
        [
            "strategy",
            "horizon",
        ]
    ):

        horizon_rows.append({

            "strategy":
                strategy,

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
        })


    horizon_summary = pd.DataFrame(
        horizon_rows
    )


    # =========================================================
    # Fixed expert summary
    # =========================================================

    fixed_rows = []


    for (
        expert,
        group,
    ) in fixed_predictions.groupby(
        "expert"
    ):

        fixed_rows.append({

            "method":
                expert,

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


    fixed_summary = (
        pd.DataFrame(
            fixed_rows
        )

        .sort_values(
            "mae"
        )

        .reset_index(
            drop=True
        )
    )


    # =========================================================
    # Combined comparison
    # =========================================================

    strategy_for_comparison = (
        summary[
            [
                "strategy",
                "mae",
                "rmse",
                "n_forecasts",
            ]
        ]
        .rename(
            columns={
                "strategy":
                    "method"
            }
        )
    )


    combined_comparison = (
        pd.concat(
            [
                strategy_for_comparison,
                fixed_summary,
            ],
            ignore_index=True,
        )

        .sort_values(
            "mae"
        )

        .reset_index(
            drop=True
        )
    )


    # =========================================================
    # Print
    # =========================================================

    print(
        "\n==================================="
    )

    print(
        "Hard Routing Comparison"
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
        "Horizon-Specific Hard Router"
    )

    print(
        "==================================="
    )


    print(
        horizon_selections.to_string(
            index=False
        )
    )


    print(
        "\n==================================="
    )

    print(
        "Horizon-Specific Performance"
    )

    print(
        "==================================="
    )


    print(
        horizon_summary
        .sort_values(
            [
                "horizon",
                "mae",
            ]
        )
        .to_string(
            index=False
        )
    )


    print(
        "\n==================================="
    )

    print(
        "Dynamic Routers vs Fixed Experts"
    )

    print(
        "==================================="
    )


    print(
        combined_comparison.to_string(
            index=False
        )
    )


    # =========================================================
    # Save
    # =========================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )


    predictions.to_csv(
        "results/"
        "nested_hard_router_predictions.csv",
        index=False,
    )


    summary.to_csv(
        "results/"
        "nested_hard_router_summary.csv",
        index=False,
    )


    horizon_summary.to_csv(
        "results/"
        "nested_hard_router_horizon_summary.csv",
        index=False,
    )


    horizon_selections.to_csv(
        "results/"
        "nested_horizon_hard_selections.csv",
        index=False,
    )


    fixed_predictions.to_csv(
        "results/"
        "nested_fixed_expert_predictions.csv",
        index=False,
    )


    combined_comparison.to_csv(
        "results/"
        "nested_hard_router_vs_fixed.csv",
        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        "results/nested_hard_router_predictions.csv"
    )

    print(
        "results/nested_hard_router_summary.csv"
    )

    print(
        "results/nested_hard_router_horizon_summary.csv"
    )

    print(
        "results/nested_horizon_hard_selections.csv"
    )

    print(
        "results/nested_hard_router_vs_fixed.csv"
    )


if __name__ == "__main__":
    main()
