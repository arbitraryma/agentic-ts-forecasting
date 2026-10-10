# LEGACY / DEBUG SCRIPT.
# Hard-codes an AirPassengers path.
# Not the config-driven paper benchmark.
# Importing this file does not launch the experiment.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import itertools

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

    from routing.state_horizon_router import (
        build_state_horizon_routing_table,
    )


    # =========================================================
    # Experiment settings
    # =========================================================

    DATA_PATH = "data/AirPassengers.csv"

    TARGET_COLUMN = "#Passengers"

    HORIZON = 3

    MAX_WINDOWS = 10

    TRAIN_WINDOW = 60

    WINDOW_TYPES = [
        "expanding",
    ]

    COMPUTE_BUDGET = "medium"

    COMPUTE_WEIGHT = 0.10


    # =========================================================
    # Sensitivity grid
    # =========================================================

    STATE_BANDWIDTHS = [
        0.5,
        1.0,
        2.0,
    ]

    TEMPERATURES = [
        0.5,
        1.0,
        2.0,
    ]

    RECENCY_DECAYS = [
        0.0,
        0.1,
        0.3,
    ]

    N_WINDOWS_GRID = [
        3,
        5,
        10,
    ]


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
    # Metadata
    # =========================================================

    n_series = 1

    has_covariates = False


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


    print(
        "\n==================================="
    )
    print(
        "Diagnostics"
    )
    print(
        "==================================="
    )

    print(
        diagnostics
    )


    # =========================================================
    # Candidate routing
    # =========================================================

    candidates, reasons = (
        route_candidates(
            diagnostics=diagnostics,
            horizon=HORIZON,
            n_series=n_series,
            has_covariates=has_covariates,
            compute_budget=COMPUTE_BUDGET,
        )
    )


    print(
        "\n==================================="
    )
    print(
        "Candidates"
    )
    print(
        "==================================="
    )

    print(
        candidates
    )


    # =========================================================
    # Rolling validation
    #
    # IMPORTANT:
    # Run models only ONCE using MAX_WINDOWS.
    # Later sensitivity analysis reuses these results.
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
            "\n==================================="
        )

        print(
            f"Running model: {model_name}"
        )

        print(
            "==================================="
        )


        try:

            model_class = (
                get_model_class(
                    model_name
                )
            )


            # Use light settings because this
            # experiment is testing routing,
            # not tuning forecasting networks.
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
                        MAX_WINDOWS,

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


                print(
                    f"{window_type}: OK"
                )


        except Exception as e:

            print(
                f"FAILED {model_name}: "
                f"{type(e).__name__}: "
                f"{e}"
            )


    # =========================================================
    # Combine results
    # =========================================================

    if len(
        all_results
    ) == 0:

        raise RuntimeError(
            "No models completed successfully."
        )


    all_results = pd.concat(
        all_results,
        ignore_index=True,
    )


    # =========================================================
    # Save raw CV results
    # =========================================================

    os.makedirs(
        "results",
        exist_ok=True,
    )


    raw_results_path = (
        "results/"
        "router_sensitivity_cv_results.csv"
    )


    all_results.to_csv(
        raw_results_path,
        index=False,
    )


    print(
        "\nSaved rolling CV results to:"
    )

    print(
        raw_results_path
    )


    # =========================================================
    # Helper:
    # keep only most recent K windows
    # for every expert
    # =========================================================

    def keep_recent_windows(
        results,
        n_windows,
    ):

        pieces = []


        groups = results.groupby(
            [
                "family",
                "model",
                "window_type",
            ],
            sort=False,
        )


        for _, group in groups:

            group = (
                group
                .sort_values(
                    "window"
                )
                .tail(
                    n_windows
                )
            )


            pieces.append(
                group
            )


        return pd.concat(
            pieces,
            ignore_index=True,
        )


    # =========================================================
    # Sensitivity experiment
    # =========================================================

    sensitivity_rows = []


    grid = itertools.product(
        STATE_BANDWIDTHS,
        TEMPERATURES,
        RECENCY_DECAYS,
        N_WINDOWS_GRID,
    )


    for (
        bandwidth,
        temperature,
        recency_decay,
        n_windows,
    ) in grid:


        # -----------------------------------------------------
        # Select last K validation origins
        # -----------------------------------------------------

        subset = keep_recent_windows(
            all_results,
            n_windows=n_windows,
        )


        # -----------------------------------------------------
        # Build dynamic routing table
        # -----------------------------------------------------

        routing_table = (
            build_state_horizon_routing_table(

                all_results=
                    subset,

                current_y=
                    y,

                horizon=
                    HORIZON,

                seasonal_period=
                    seasonal_period,

                compute_weight=
                    COMPUTE_WEIGHT,

                temperature=
                    temperature,

                state_bandwidth=
                    bandwidth,

                recency_decay=
                    recency_decay,
            )
        )


        # -----------------------------------------------------
        # Record winner + weight structure
        # horizon by horizon
        # -----------------------------------------------------

        for h in range(
            1,
            HORIZON + 1,
        ):

            table_h = (
                routing_table[
                    routing_table[
                        "horizon"
                    ]
                    == h
                ]
                .sort_values(
                    "routing_weight",
                    ascending=False,
                )
                .reset_index(
                    drop=True
                )
            )


            winner = (
                table_h.iloc[
                    0
                ]
            )


            # -----------------------------------------------
            # Routing entropy
            #
            # Low entropy:
            # very concentrated weights
            #
            # High entropy:
            # diversified ensemble
            # -----------------------------------------------

            weights = (
                table_h[
                    "routing_weight"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            positive_weights = (
                weights[
                    weights > 0
                ]
            )


            entropy = float(
                -np.sum(
                    positive_weights
                    * np.log(
                        positive_weights
                    )
                )
            )


            # -----------------------------------------------
            # Maximum routing weight
            # -----------------------------------------------

            max_weight = float(
                np.max(
                    weights
                )
            )


            # -----------------------------------------------
            # Effective sample size
            # from historical localization
            # -----------------------------------------------

            mean_ess = float(
                table_h[
                    "effective_sample_size"
                ]
                .mean()
            )


            sensitivity_rows.append({

                "state_bandwidth":
                    bandwidth,

                "temperature":
                    temperature,

                "recency_decay":
                    recency_decay,

                "n_windows":
                    n_windows,

                "horizon":
                    h,

                "winner_model":
                    winner[
                        "model"
                    ],

                "winner_family":
                    winner[
                        "family"
                    ],

                "winner_window_type":
                    winner[
                        "window_type"
                    ],

                "winner_weight":
                    float(
                        winner[
                            "routing_weight"
                        ]
                    ),

                "winner_loss":
                    float(
                        winner[
                            "localized_loss"
                        ]
                    ),

                "routing_entropy":
                    entropy,

                "max_weight":
                    max_weight,

                "mean_effective_sample_size":
                    mean_ess,
            })


    # =========================================================
    # Sensitivity table
    # =========================================================

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )


    output_path = (
        "results/"
        "router_sensitivity_summary.csv"
    )


    sensitivity.to_csv(
        output_path,
        index=False,
    )


    print(
        "\n==================================="
    )
    print(
        "Sensitivity Results"
    )
    print(
        "==================================="
    )

    print(
        sensitivity
    )


    print(
        "\nSaved sensitivity results to:"
    )

    print(
        output_path
    )


    # =========================================================
    # Winner stability
    # =========================================================

    winner_stability = (
        sensitivity
        .groupby(
            [
                "horizon",
                "winner_model",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


    winner_stability[
        "proportion"
    ] = (
        winner_stability[
            "count"
        ]
        /
        winner_stability
        .groupby(
            "horizon"
        )[
            "count"
        ]
        .transform(
            "sum"
        )
    )


    winner_stability = (
        winner_stability
        .sort_values(
            [
                "horizon",
                "proportion",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


    print(
        "\n==================================="
    )
    print(
        "Winner Stability"
    )
    print(
        "==================================="
    )

    print(
        winner_stability
    )


    winner_path = (
        "results/"
        "router_winner_stability.csv"
    )


    winner_stability.to_csv(
        winner_path,
        index=False,
    )


    # =========================================================
    # Average routing sharpness
    # =========================================================

    sharpness_summary = (
        sensitivity

        .groupby(
            [
                "state_bandwidth",
                "temperature",
                "recency_decay",
                "n_windows",
            ]
        )[
            [
                "max_weight",
                "routing_entropy",
                "mean_effective_sample_size",
            ]
        ]

        .mean()

        .reset_index()

        .sort_values(
            "max_weight",
            ascending=False,
        )
    )


    print(
        "\n==================================="
    )
    print(
        "Routing Sharpness Summary"
    )
    print(
        "==================================="
    )

    print(
        sharpness_summary
    )


    sharpness_path = (
        "results/"
        "router_sharpness_summary.csv"
    )


    sharpness_summary.to_csv(
        sharpness_path,
        index=False,
    )


    print(
        "\n==================================="
    )
    print(
        "Finished"
    )
    print(
        "==================================="
    )

    print(
        "Created:"
    )

    print(
        raw_results_path
    )

    print(
        output_path
    )

    print(
        winner_path
    )

    print(
        sharpness_path
    )


if __name__ == "__main__":
    main()
