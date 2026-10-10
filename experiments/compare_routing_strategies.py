# LEGACY / DEBUG SCRIPT.
# Not the paper soft-routing ablation.
# This file keeps the top 3 experts and renormalizes
# their routing weights.
# The paper ablation is the full softmax in
# evaluation/nested_evaluation.py (SOFT_ROUTING = "full_softmax"),
# with no top-k truncation.
# Importing this file does not launch the experiment.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from diagnostics.basic import (
    basic_diagnostics,
)


def main():

    from evaluation.rolling_cv import (
        rolling_backtest,
    )

    from models.registry import (
        MODEL_REGISTRY,
        get_model_class,
    )

    from models.config_builder import (
        build_model_kwargs,
    )

    from routing.candidate_router import (
        route_candidates,
    )

    from routing.validation_router import (
        compute_routing_scores,
    )

    from routing.horizon_router import (
        build_horizon_routing_table,
    )

    from routing.state_horizon_router import (
        build_state_horizon_routing_table,
    )


    # =========================================================
    # Experiment settings
    # =========================================================

    DATA_PATH = "data/AirPassengers.csv"
    TARGET_COLUMN = "#Passengers"

    HORIZON = 3

    N_WINDOWS = 10

    TRAIN_WINDOW = 60

    WINDOW_TYPES = [
        "expanding",
    ]

    COMPUTE_BUDGET = "medium"

    COMPUTE_WEIGHT = 0.10

    TEMPERATURE = 1.0

    STATE_BANDWIDTH = 2.0

    RECENCY_DECAY = 0.1

    TOP_K = 3


    # =========================================================
    # Metrics
    # =========================================================

    def mae(
        y_true,
        y_pred,
    ):
        y_true = np.asarray(
            y_true,
            dtype=float,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float,
        )

        return float(
            np.mean(
                np.abs(
                    y_true - y_pred
                )
            )
        )


    def rmse(
        y_true,
        y_pred,
    ):
        y_true = np.asarray(
            y_true,
            dtype=float,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float,
        )

        return float(
            np.sqrt(
                np.mean(
                    (
                        y_true - y_pred
                    ) ** 2
                )
            )
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
        .reset_index(drop=True)
    )


    # =========================================================
    # Diagnostics
    # =========================================================

    diagnostics = (
        basic_diagnostics(
            y
        )
    )

    seasonal_period = (
        diagnostics.get(
            "seasonal_period"
        )
    )


    n_series = 1

    has_covariates = False


    # =========================================================
    # Candidate routing
    # =========================================================

    candidates, _ = (
        route_candidates(
            diagnostics=diagnostics,
            horizon=HORIZON,
            n_series=n_series,
            has_covariates=has_covariates,
            compute_budget=COMPUTE_BUDGET,
        )
    )


    print(
        "\nCandidates:"
    )

    print(
        candidates
    )


    # =========================================================
    # Run rolling validation
    # =========================================================

    all_results = []


    for model_name in candidates:

        config = MODEL_REGISTRY[
            model_name
        ]

        if not config[
            "enabled"
        ]:
            continue


        print(
            f"\nRunning {model_name}"
        )


        try:

            model_class = (
                get_model_class(
                    model_name
                )
            )


            model_kwargs = (
                build_model_kwargs(
                    model_name=model_name,
                    diagnostics=diagnostics,
                    fast=True,
                    horizon=HORIZON,
                    n_series=n_series,
                )
            )


            for window_type in WINDOW_TYPES:

                kwargs = {

                    "model_class":
                        model_class,

                    "y":
                        y,

                    "horizon":
                        HORIZON,

                    "n_windows":
                        N_WINDOWS,

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


                results = (
                    rolling_backtest(
                        **kwargs
                    )
                )


                results[
                    "model"
                ] = model_name


                results[
                    "family"
                ] = config[
                    "family"
                ]


                all_results.append(
                    results
                )


        except Exception as e:

            print(
                f"FAILED {model_name}: "
                f"{type(e).__name__}: {e}"
            )


    if len(
        all_results
    ) == 0:

        raise RuntimeError(
            "No successful model results."
        )


    all_results = pd.concat(
        all_results,
        ignore_index=True,
    )


    # =========================================================
    # Overall routing table
    # =========================================================

    overall_summary = (
        all_results

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


    # =========================================================
    # Horizon-specific table
    # =========================================================

    horizon_routing = (
        build_horizon_routing_table(
            all_results=
                all_results,

            horizon=
                HORIZON,

            compute_weight=
                COMPUTE_WEIGHT,

            temperature=
                TEMPERATURE,
        )
    )


    # =========================================================
    # State+horizon routing
    # =========================================================

    state_horizon_routing = (
        build_state_horizon_routing_table(

            all_results=
                all_results,

            current_y=
                y,

            horizon=
                HORIZON,

            seasonal_period=
                seasonal_period,

            compute_weight=
                COMPUTE_WEIGHT,

            temperature=
                TEMPERATURE,

            state_bandwidth=
                STATE_BANDWIDTH,

            recency_decay=
                RECENCY_DECAY,
        )
    )


    # =========================================================
    # Select top-K overall experts
    # =========================================================

    top_experts = (
        overall_routing

        .sort_values(
            "utility",
            ascending=False,
        )

        .head(
            TOP_K
        )

        .reset_index(
            drop=True
        )
    )


    print(
        "\nTop experts:"
    )

    print(
        top_experts[
            [
                "model",
                "window_type",
                "mae",
                "routing_weight",
            ]
        ]
    )


    # =========================================================
    # Build final expert forecasts
    # =========================================================

    expert_forecasts = []

    expert_names = []

    expert_models = []

    expert_window_types = []


    for _, row in top_experts.iterrows():

        model_name = row[
            "model"
        ]

        window_type = row[
            "window_type"
        ]


        model_class = (
            get_model_class(
                model_name
            )
        )


        model_kwargs = (
            build_model_kwargs(
                model_name=model_name,
                diagnostics=diagnostics,
                fast=True,
                horizon=HORIZON,
                n_series=n_series,
            )
        )


        if window_type == "sliding":

            train_y = (
                y.iloc[
                    -TRAIN_WINDOW:
                ]
                .reset_index(
                    drop=True
                )
            )

        else:

            train_y = (
                y.copy()
            )


        model = model_class(
            **model_kwargs
        )

        model.fit(
            train_y
        )


        forecast = model.predict(
            horizon=HORIZON
        )


        forecast = np.asarray(
            forecast,
            dtype=float,
        )

        forecast = np.squeeze(
            forecast
        )


        expert_forecasts.append(
            forecast
        )

        expert_names.append(
            f"{model_name}__{window_type}"
        )

        expert_models.append(
            model_name
        )

        expert_window_types.append(
            window_type
        )


    expert_forecasts = np.asarray(
        expert_forecasts,
        dtype=float,
    )


    # =========================================================
    # Strategy 1:
    # Best single model
    # =========================================================

    best_forecast = (
        expert_forecasts[
            0
        ]
    )


    # =========================================================
    # Strategy 2:
    # Equal-weight ensemble
    # =========================================================

    equal_weights = (
        np.ones(
            len(
                expert_forecasts
            )
        )
        /
        len(
            expert_forecasts
        )
    )


    equal_forecast = (
        np.sum(
            expert_forecasts
            * equal_weights[
                :,
                None
            ],
            axis=0,
        )
    )


    # =========================================================
    # Strategy 3:
    # Overall routing weights w_m
    # =========================================================

    overall_weights = (
        top_experts[
            "routing_weight"
        ]
        .to_numpy(
            dtype=float
        )
    )


    overall_weights = (
        overall_weights
        /
        overall_weights.sum()
    )


    overall_forecast = (
        np.sum(
            expert_forecasts
            * overall_weights[
                :,
                None
            ],
            axis=0,
        )
    )


    # =========================================================
    # Strategy 4:
    # Horizon-specific weights w_{m,h}
    # =========================================================

    horizon_forecast = []


    for h in range(
        1,
        HORIZON + 1,
    ):

        weights_h = []


        for (
            model_name,
            window_type,
        ) in zip(
            expert_models,
            expert_window_types,
        ):

            match = (
                horizon_routing[
                    (
                        horizon_routing[
                            "model"
                        ]
                        == model_name
                    )
                    &
                    (
                        horizon_routing[
                            "window_type"
                        ]
                        == window_type
                    )
                    &
                    (
                        horizon_routing[
                            "horizon"
                        ]
                        == h
                    )
                ]
            )


            if len(match) == 0:

                weights_h.append(
                    0.0
                )

            else:

                weights_h.append(
                    float(
                        match.iloc[
                            0
                        ][
                            "routing_weight"
                        ]
                    )
                )


        weights_h = np.asarray(
            weights_h,
            dtype=float,
        )


        if weights_h.sum() <= 0:

            weights_h = (
                np.ones_like(
                    weights_h
                )
                /
                len(
                    weights_h
                )
            )

        else:

            weights_h = (
                weights_h
                /
                weights_h.sum()
            )


        pred_h = float(
            np.sum(
                expert_forecasts[
                    :,
                    h - 1
                ]
                * weights_h
            )
        )


        horizon_forecast.append(
            pred_h
        )


    horizon_forecast = np.asarray(
        horizon_forecast
    )


    # =========================================================
    # Strategy 5:
    # State+horizon weights w_{m,h,t}
    # =========================================================

    state_forecast = []


    for h in range(
        1,
        HORIZON + 1,
    ):

        weights_h = []


        for (
            model_name,
            window_type,
        ) in zip(
            expert_models,
            expert_window_types,
        ):

            match = (
                state_horizon_routing[
                    (
                        state_horizon_routing[
                            "model"
                        ]
                        == model_name
                    )
                    &
                    (
                        state_horizon_routing[
                            "window_type"
                        ]
                        == window_type
                    )
                    &
                    (
                        state_horizon_routing[
                            "horizon"
                        ]
                        == h
                    )
                ]
            )


            if len(match) == 0:

                weights_h.append(
                    0.0
                )

            else:

                weights_h.append(
                    float(
                        match.iloc[
                            0
                        ][
                            "routing_weight"
                        ]
                    )
                )


        weights_h = np.asarray(
            weights_h,
            dtype=float,
        )


        if weights_h.sum() <= 0:

            weights_h = (
                np.ones_like(
                    weights_h
                )
                /
                len(
                    weights_h
                )
            )

        else:

            weights_h = (
                weights_h
                /
                weights_h.sum()
            )


        pred_h = float(
            np.sum(
                expert_forecasts[
                    :,
                    h - 1
                ]
                * weights_h
            )
        )


        state_forecast.append(
            pred_h
        )


    state_forecast = np.asarray(
        state_forecast
    )


    # =========================================================
    # Display forecasts
    # =========================================================

    forecast_table = pd.DataFrame({

        "horizon":
            np.arange(
                1,
                HORIZON + 1
            ),

        "best_single":
            best_forecast,

        "equal_weight":
            equal_forecast,

        "overall_weight":
            overall_forecast,

        "horizon_weight":
            horizon_forecast,

        "state_horizon_weight":
            state_forecast,
    })


    print(
        "\n==================================="
    )
    print(
        "Final Forecast Comparison"
    )
    print(
        "==================================="
    )

    print(
        forecast_table
    )


    # =========================================================
    # Save forecasts
    # =========================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )


    forecast_table.to_csv(
        "results/routing_strategy_forecasts.csv",
        index=False,
    )


    print(
        "\nSaved:"
    )

    print(
        "results/routing_strategy_forecasts.csv"
    )


if __name__ == "__main__":
    main()
