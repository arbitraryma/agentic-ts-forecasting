# LEGACY / DEBUG SCRIPT.
# Hard-codes an AirPassengers path.
# Not the config-driven paper benchmark.
# Importing this file does not launch the experiment.

import argparse
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

    from diagnostics.state import (
        extract_state_features,
    )

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
        select_top_models,
    )

    from routing.horizon_router import (
        build_horizon_routing_table,
        get_horizon_weights,
    )

    from routing.state_horizon_router import (
        build_state_horizon_routing_table,
    )


    # ===================================
    # Command-line arguments
    # ===================================

    parser = argparse.ArgumentParser(
        description=(
            "Candidate-routed, validation-driven, "
            "state-aware adaptive forecasting system."
        )
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run lightweight benchmark mode.",
    )

    parser.add_argument(
        "--compute-weight",
        type=float,
        default=0.10,
        help="Runtime penalty used in routing utility.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.50,
        help="Softmax temperature used for routing weights.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of experts retained for final ensemble.",
    )

    parser.add_argument(
        "--compute-budget",
        type=str,
        default="medium",
        choices=[
            "low",
            "medium",
            "high",
        ],
        help="Compute budget used by candidate router.",
    )

    parser.add_argument(
        "--state-bandwidth",
        type=float,
        default=1.0,
        help=(
            "Bandwidth controlling how strongly "
            "state similarity localizes validation."
        ),
    )

    parser.add_argument(
        "--recency-decay",
        type=float,
        default=0.0,
        help=(
            "Exponential recency decay for historical "
            "validation windows."
        ),
    )

    args = parser.parse_args()


    # ===================================
    # Load dataset
    # ===================================

    df = pd.read_csv(
        "data/AirPassengers.csv"
    )

    y = (
        df["#Passengers"]
        .astype(float)
        .reset_index(drop=True)
    )


    # ===================================
    # Dataset metadata
    # ===================================

    n_series = 1

    has_covariates = False


    # ===================================
    # Diagnostics
    # ===================================

    diagnostics = basic_diagnostics(
        y
    )

    seasonal_period = diagnostics.get(
        "seasonal_period"
    )


    print(
        "\n==================================="
    )
    print(
        "Diagnostics"
    )
    print(
        "==================================="
    )

    for key, value in diagnostics.items():
        print(
            f"{key}: {value}"
        )


    # ===================================
    # Current state
    # ===================================

    current_state = extract_state_features(
        y,
        seasonal_period=seasonal_period,
    )


    print(
        "\n==================================="
    )
    print(
        "Current Time-Series State"
    )
    print(
        "==================================="
    )

    for key, value in current_state.items():

        print(
            f"{key}: {value:.4f}"
        )


    # ===================================
    # Backtest settings
    # ===================================

    if args.fast:

        print(
            "\nFAST MODE ENABLED"
        )

        horizon = 3

        # For state-aware routing, 1 window is
        # not informative. We still keep fast
        # mode small, but use several windows.
        n_windows = 3

        train_window = 36

        window_types = [
            "expanding",
        ]

    else:

        print(
            "\nFULL MODE ENABLED"
        )

        horizon = 12

        n_windows = 6

        train_window = 60

        window_types = [
            "expanding",
            "sliding",
        ]


    print(
        "\nBenchmark settings:"
    )

    print(
        f"horizon = {horizon}"
    )

    print(
        f"n_windows = {n_windows}"
    )

    print(
        f"window_types = {window_types}"
    )

    print(
        f"compute_budget = {args.compute_budget}"
    )

    print(
        f"state_bandwidth = {args.state_bandwidth}"
    )

    print(
        f"recency_decay = {args.recency_decay}"
    )


    # ===================================
    # Candidate router
    # ===================================

    candidates, candidate_reasons = (
        route_candidates(
            diagnostics=diagnostics,
            horizon=horizon,
            n_series=n_series,
            has_covariates=has_covariates,
            compute_budget=args.compute_budget,
        )
    )


    print(
        "\n==================================="
    )
    print(
        "Candidate Router"
    )
    print(
        "==================================="
    )


    if len(candidates) == 0:

        raise RuntimeError(
            "Candidate router returned no models."
        )


    for model_name in candidates:

        print(
            f"\n{model_name}"
        )

        for reason in candidate_reasons[
            model_name
        ]:

            print(
                f"  - {reason}"
            )


    print(
        "\nSelected candidate count:",
        len(candidates),
    )

    print(
        "Candidates:",
        candidates,
    )


    # ===================================
    # Rolling validation
    # ===================================

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
            "\n==================================="
        )

        print(
            f"Running model: {model_name}"
        )

        print(
            "==================================="
        )


        # ===================================
        # Lazy load model
        # ===================================

        try:

            model_class = (
                get_model_class(
                    model_name
                )
            )

        except Exception as e:

            print(
                f"Could not load "
                f"{model_name}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            continue


        # ===================================
        # Model configuration
        # ===================================

        try:

            model_kwargs = (
                build_model_kwargs(
                    model_name=model_name,
                    diagnostics=diagnostics,
                    fast=args.fast,
                    horizon=horizon,
                    n_series=n_series,
                )
            )

        except Exception as e:

            print(
                f"Could not build config "
                f"for {model_name}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            continue


        print(
            "Model parameters:"
        )

        print(
            model_kwargs
        )


        # ===================================
        # Rolling-origin evaluation
        # ===================================

        for window_type in window_types:

            print(
                f"\nRunning {window_type} CV..."
            )


            try:

                backtest_kwargs = {

                    "model_class":
                        model_class,

                    "y":
                        y,

                    "horizon":
                        horizon,

                    "n_windows":
                        n_windows,

                    "model_kwargs":
                        model_kwargs,

                    "window_type":
                        window_type,

                    "seasonal_period":
                        seasonal_period,
                }


                if window_type == "sliding":

                    backtest_kwargs[
                        "train_window"
                    ] = train_window


                results = (
                    rolling_backtest(
                        **backtest_kwargs
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


                print(
                    f"{window_type} CV: OK"
                )


            except Exception as e:

                print(
                    f"{window_type} CV "
                    f"FAILED for "
                    f"{model_name}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )


    # ===================================
    # Check successful models
    # ===================================

    if len(
        all_results
    ) == 0:

        raise RuntimeError(
            "No candidate model completed successfully."
        )


    # ===================================
    # Combine validation results
    # ===================================

    all_results = pd.concat(
        all_results,
        ignore_index=True,
    )


    print(
        "\n==================================="
    )
    print(
        "All CV Results"
    )
    print(
        "==================================="
    )

    print(
        all_results
    )


    # ===================================
    # Overall validation summary
    # ===================================

    summary = (
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


    print(
        "\n==================================="
    )
    print(
        "Overall Model Comparison"
    )
    print(
        "==================================="
    )

    print(
        summary
    )


    # ===================================
    # Overall validation-driven routing
    # ===================================

    routing_table = (
        compute_routing_scores(
            validation_summary=summary,
            metric="mae",
            compute_weight=args.compute_weight,
            temperature=args.temperature,
        )
    )


    print(
        "\n==================================="
    )
    print(
        "Overall Validation-Driven Routing"
    )
    print(
        "==================================="
    )


    print(
        routing_table[
            [
                "family",
                "model",
                "window_type",
                "mae",
                "runtime",
                "utility",
                "routing_weight",
                "routing_rank",
            ]
        ]
    )


    # ===================================
    # Non-state horizon routing
    # ===================================

    horizon_routing_table = (
        build_horizon_routing_table(
            all_results=all_results,
            horizon=horizon,
            compute_weight=args.compute_weight,
            temperature=args.temperature,
        )
    )


    # ===================================
    # State-aware horizon routing
    # ===================================

    state_horizon_table = (
        build_state_horizon_routing_table(

            all_results=
                all_results,

            current_y=
                y,

            horizon=
                horizon,

            seasonal_period=
                seasonal_period,

            compute_weight=
                args.compute_weight,

            temperature=
                args.temperature,

            state_bandwidth=
                args.state_bandwidth,

            recency_decay=
                args.recency_decay,
        )
    )


    print(
        "\n==================================="
    )
    print(
        "State-Aware Horizon Routing"
    )
    print(
        "==================================="
    )


    for h in range(
        1,
        horizon + 1,
    ):

        print(
            f"\n--- Horizon h={h} ---"
        )


        table_h = (
            state_horizon_table[
                state_horizon_table[
                    "horizon"
                ]
                == h
            ]
            .sort_values(
                "utility",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        print(
            table_h[
                [
                    "model",
                    "family",
                    "window_type",
                    "localized_loss",
                    "runtime",
                    "effective_sample_size",
                    "utility",
                    "routing_weight",
                    "routing_rank",
                ]
            ]
        )


    # ===================================
    # Select overall top-k experts
    # ===================================

    top_k = min(
        args.top_k,
        len(routing_table),
    )


    top_experts = (
        select_top_models(
            routing_table,
            top_k=top_k,
        )
    )


    print(
        "\n==================================="
    )
    print(
        f"Top {top_k} Overall Experts"
    )
    print(
        "==================================="
    )


    print(
        top_experts[
            [
                "model",
                "family",
                "window_type",
                "mae",
                "runtime",
                "utility",
                "routing_weight",
            ]
        ]
    )


    # ===================================
    # Generate final expert forecasts
    # ===================================

    expert_forecasts = []

    expert_names = []

    expert_models = []

    expert_window_types = []


    print(
        "\n==================================="
    )
    print(
        "Generating Final Expert Forecasts"
    )
    print(
        "==================================="
    )


    for _, row in top_experts.iterrows():

        model_name = row[
            "model"
        ]

        window_type = row[
            "window_type"
        ]


        expert_name = (
            f"{model_name}"
            f"__"
            f"{window_type}"
        )


        print(
            f"\nExpert: {expert_name}"
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
                    fast=args.fast,
                    horizon=horizon,
                    n_series=n_series,
                )
            )


            # ===================================
            # Training history
            # ===================================

            if window_type == "sliding":

                train_y = (
                    y.iloc[
                        -train_window:
                    ]
                    .reset_index(
                        drop=True
                    )
                )

            else:

                train_y = (
                    y.copy()
                )


            # ===================================
            # Fit
            # ===================================

            model = model_class(
                **model_kwargs
            )


            model.fit(
                train_y
            )


            # ===================================
            # Forecast
            # ===================================

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
                    f"Expected 1D forecast, "
                    f"got shape "
                    f"{forecast.shape}."
                )


            if len(
                forecast
            ) != horizon:

                raise ValueError(
                    "Forecast length does not "
                    "match horizon."
                )


            expert_forecasts.append(
                forecast
            )

            expert_names.append(
                expert_name
            )

            expert_models.append(
                model_name
            )

            expert_window_types.append(
                window_type
            )


            print(
                "Forecast:"
            )

            print(
                forecast
            )


        except Exception as e:

            print(
                f"Final forecast FAILED "
                f"for {expert_name}: "
                f"{type(e).__name__}: "
                f"{e}"
            )


    # ===================================
    # Check final experts
    # ===================================

    if len(
        expert_forecasts
    ) == 0:

        raise RuntimeError(
            "No expert produced a final forecast."
        )


    expert_forecasts = np.asarray(
        expert_forecasts,
        dtype=float,
    )


    # ===================================
    # State-aware horizon ensemble
    # ===================================

    final_forecast = []

    weight_rows = []


    for h in range(
        1,
        horizon + 1,
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
                state_horizon_table[
                    (
                        state_horizon_table[
                            "model"
                        ]
                        == model_name
                    )
                    &
                    (
                        state_horizon_table[
                            "window_type"
                        ]
                        == window_type
                    )
                    &
                    (
                        state_horizon_table[
                            "horizon"
                        ]
                        == h
                    )
                ]
            )


            if len(match) == 0:

                weight = 0.0

            else:

                weight = float(
                    match.iloc[
                        0
                    ][
                        "routing_weight"
                    ]
                )


            weights_h.append(
                weight
            )


        weights_h = np.asarray(
            weights_h,
            dtype=float,
        )


        # ===================================
        # Renormalize over selected top-k
        # ===================================

        total_weight = float(
            np.sum(
                weights_h
            )
        )


        if total_weight <= 0:

            weights_h = (
                np.ones(
                    len(
                        expert_forecasts
                    )
                )
                / len(
                    expert_forecasts
                )
            )

        else:

            weights_h = (
                weights_h
                / total_weight
            )


        # ===================================
        # Ensemble prediction at horizon h
        # ===================================

        forecast_h = float(
            np.sum(
                expert_forecasts[
                    :,
                    h - 1
                ]
                * weights_h
            )
        )


        final_forecast.append(
            forecast_h
        )


        for (
            expert_name,
            weight,
        ) in zip(
            expert_names,
            weights_h,
        ):

            weight_rows.append({

                "horizon":
                    h,

                "expert":
                    expert_name,

                "weight":
                    float(
                        weight
                    ),
            })


    final_forecast = np.asarray(
        final_forecast,
        dtype=float,
    )


    weight_table = pd.DataFrame(
        weight_rows
    )


    # ===================================
    # Print state-aware weights
    # ===================================

    print(
        "\n==================================="
    )
    print(
        "Final State-Aware "
        "Horizon-Specific Ensemble Weights"
    )
    print(
        "==================================="
    )


    for h in range(
        1,
        horizon + 1,
    ):

        print(
            f"\n--- Horizon h={h} ---"
        )


        table_h = (
            weight_table[
                weight_table[
                    "horizon"
                ]
                == h
            ]
            .sort_values(
                "weight",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        print(
            table_h[
                [
                    "expert",
                    "weight",
                ]
            ]
        )


    # ===================================
    # Final dynamic forecast
    # ===================================

    print(
        "\n==================================="
    )
    print(
        "Final State-Aware Dynamic Forecast"
    )
    print(
        "==================================="
    )


    for h, prediction in enumerate(
        final_forecast,
        start=1,
    ):

        print(
            f"t+{h}: "
            f"{prediction:.4f}"
        )


    # ===================================
    # Best hard-routing expert
    # ===================================

    best_expert = (
        routing_table.iloc[
            0
        ]
    )


    print(
        "\n==================================="
    )
    print(
        "Best Overall Hard-Routing Expert"
    )
    print(
        "==================================="
    )

    print(
        f"Model: "
        f"{best_expert['model']}"
    )

    print(
        f"Family: "
        f"{best_expert['family']}"
    )

    print(
        f"Window type: "
        f"{best_expert['window_type']}"
    )

    print(
        f"MAE: "
        f"{best_expert['mae']:.4f}"
    )

    print(
        f"Runtime: "
        f"{best_expert['runtime']:.4f} sec"
    )

    print(
        f"Utility: "
        f"{best_expert['utility']:.4f}"
    )

    print(
        f"Routing weight: "
        f"{best_expert['routing_weight']:.4f}"
    )


if __name__ == "__main__":
    main()
