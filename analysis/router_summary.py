import os
import pandas as pd


def summarize_router_sensitivity(
    results_dir="results",
):
    """
    Read router sensitivity CSV outputs and return
    a compact structured summary for downstream
    LLM reasoning.

    Expected files
    --------------
    router_sensitivity_summary.csv
    router_winner_stability.csv
    router_sharpness_summary.csv

    Returns
    -------
    dict
        JSON-serializable summary.
    """

    sensitivity_path = os.path.join(
        results_dir,
        "router_sensitivity_summary.csv",
    )

    winner_path = os.path.join(
        results_dir,
        "router_winner_stability.csv",
    )

    sharpness_path = os.path.join(
        results_dir,
        "router_sharpness_summary.csv",
    )


    # ===================================
    # Validate files
    # ===================================

    required_files = [
        sensitivity_path,
        winner_path,
        sharpness_path,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )


    # ===================================
    # Load tables
    # ===================================

    sensitivity = pd.read_csv(
        sensitivity_path
    )

    winner_stability = pd.read_csv(
        winner_path
    )

    sharpness = pd.read_csv(
        sharpness_path
    )


    # ===================================
    # Winner stability by horizon
    # ===================================

    winner_summary = {}


    for horizon in sorted(
        winner_stability[
            "horizon"
        ].unique()
    ):

        table_h = (
            winner_stability[
                winner_stability[
                    "horizon"
                ]
                == horizon
            ]
            .sort_values(
                "proportion",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        winner_summary[
            f"h{int(horizon)}"
        ] = [
            {
                "model":
                    str(
                        row[
                            "winner_model"
                        ]
                    ),

                "proportion":
                    float(
                        row[
                            "proportion"
                        ]
                    ),

                "count":
                    int(
                        row[
                            "count"
                        ]
                    ),
            }

            for _, row
            in table_h.iterrows()
        ]


    # ===================================
    # Temperature effect
    # ===================================

    temperature_summary = (
        sensitivity

        .groupby(
            "temperature"
        )[
            [
                "winner_weight",
                "routing_entropy",
                "max_weight",
                "mean_effective_sample_size",
            ]
        ]

        .mean()

        .reset_index()
    )


    temperature_effect = []

    for _, row in temperature_summary.iterrows():

        temperature_effect.append({

            "temperature":
                float(
                    row[
                        "temperature"
                    ]
                ),

            "mean_winner_weight":
                float(
                    row[
                        "winner_weight"
                    ]
                ),

            "mean_max_weight":
                float(
                    row[
                        "max_weight"
                    ]
                ),

            "mean_routing_entropy":
                float(
                    row[
                        "routing_entropy"
                    ]
                ),

            "mean_effective_sample_size":
                float(
                    row[
                        "mean_effective_sample_size"
                    ]
                ),
        })


    # ===================================
    # State bandwidth effect
    # ===================================

    bandwidth_summary = (
        sensitivity

        .groupby(
            "state_bandwidth"
        )[
            [
                "winner_weight",
                "routing_entropy",
                "max_weight",
                "mean_effective_sample_size",
            ]
        ]

        .mean()

        .reset_index()
    )


    bandwidth_effect = []

    for _, row in bandwidth_summary.iterrows():

        bandwidth_effect.append({

            "state_bandwidth":
                float(
                    row[
                        "state_bandwidth"
                    ]
                ),

            "mean_winner_weight":
                float(
                    row[
                        "winner_weight"
                    ]
                ),

            "mean_max_weight":
                float(
                    row[
                        "max_weight"
                    ]
                ),

            "mean_routing_entropy":
                float(
                    row[
                        "routing_entropy"
                    ]
                ),

            "mean_effective_sample_size":
                float(
                    row[
                        "mean_effective_sample_size"
                    ]
                ),
        })


    # ===================================
    # Recency effect
    # ===================================

    recency_summary = (
        sensitivity

        .groupby(
            "recency_decay"
        )[
            [
                "winner_weight",
                "routing_entropy",
                "max_weight",
                "mean_effective_sample_size",
            ]
        ]

        .mean()

        .reset_index()
    )


    recency_effect = []

    for _, row in recency_summary.iterrows():

        recency_effect.append({

            "recency_decay":
                float(
                    row[
                        "recency_decay"
                    ]
                ),

            "mean_winner_weight":
                float(
                    row[
                        "winner_weight"
                    ]
                ),

            "mean_max_weight":
                float(
                    row[
                        "max_weight"
                    ]
                ),

            "mean_routing_entropy":
                float(
                    row[
                        "routing_entropy"
                    ]
                ),

            "mean_effective_sample_size":
                float(
                    row[
                        "mean_effective_sample_size"
                    ]
                ),
        })


    # ===================================
    # Number-of-windows effect
    # ===================================

    window_summary = (
        sensitivity

        .groupby(
            "n_windows"
        )[
            [
                "winner_weight",
                "routing_entropy",
                "max_weight",
                "mean_effective_sample_size",
            ]
        ]

        .mean()

        .reset_index()
    )


    n_windows_effect = []

    for _, row in window_summary.iterrows():

        n_windows_effect.append({

            "n_windows":
                int(
                    row[
                        "n_windows"
                    ]
                ),

            "mean_winner_weight":
                float(
                    row[
                        "winner_weight"
                    ]
                ),

            "mean_max_weight":
                float(
                    row[
                        "max_weight"
                    ]
                ),

            "mean_routing_entropy":
                float(
                    row[
                        "routing_entropy"
                    ]
                ),

            "mean_effective_sample_size":
                float(
                    row[
                        "mean_effective_sample_size"
                    ]
                ),
        })


    # ===================================
    # Candidate configuration ranking
    # ===================================

    candidate_configs = (
        sharpness
        .copy()
    )


    # We want:
    # - not too concentrated
    # - reasonably high entropy
    # - reasonably high ESS
    #
    # This is NOT the final scientific criterion.
    # It is just a compact descriptive score
    # for LLM interpretation.

    max_weight_rank = (
        candidate_configs[
            "max_weight"
        ]
        .rank(
            ascending=True,
            pct=True,
        )
    )


    entropy_rank = (
        candidate_configs[
            "routing_entropy"
        ]
        .rank(
            ascending=False,
            pct=True,
        )
    )


    ess_rank = (
        candidate_configs[
            "mean_effective_sample_size"
        ]
        .rank(
            ascending=False,
            pct=True,
        )
    )


    candidate_configs[
        "descriptive_stability_score"
    ] = (
        max_weight_rank
        + entropy_rank
        + ess_rank
    ) / 3.0


    candidate_configs = (
        candidate_configs
        .sort_values(
            "descriptive_stability_score",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )


    top_configs = []


    for _, row in (
        candidate_configs
        .head(10)
        .iterrows()
    ):

        top_configs.append({

            "state_bandwidth":
                float(
                    row[
                        "state_bandwidth"
                    ]
                ),

            "temperature":
                float(
                    row[
                        "temperature"
                    ]
                ),

            "recency_decay":
                float(
                    row[
                        "recency_decay"
                    ]
                ),

            "n_windows":
                int(
                    row[
                        "n_windows"
                    ]
                ),

            "mean_max_weight":
                float(
                    row[
                        "max_weight"
                    ]
                ),

            "mean_routing_entropy":
                float(
                    row[
                        "routing_entropy"
                    ]
                ),

            "mean_effective_sample_size":
                float(
                    row[
                        "mean_effective_sample_size"
                    ]
                ),

            "descriptive_stability_score":
                float(
                    row[
                        "descriptive_stability_score"
                    ]
                ),
        })


    # ===================================
    # Basic warnings
    # ===================================

    warnings = []


    if (
        sensitivity[
            "max_weight"
        ].mean()
        > 0.90
    ):

        warnings.append(
            "Average routing weights are highly concentrated."
        )


    if (
        sensitivity[
            "mean_effective_sample_size"
        ].mean()
        < 2.0
    ):

        warnings.append(
            "Average effective sample size is very low."
        )


    if (
        sensitivity[
            "n_windows"
        ].max()
        < 5
    ):

        warnings.append(
            "Sensitivity analysis uses few rolling origins."
        )


    # ===================================
    # Final structured summary
    # ===================================

    summary = {

        "experiment": {
            "n_sensitivity_rows":
                int(
                    len(
                        sensitivity
                    )
                ),

            "horizons":
                [
                    int(x)
                    for x in sorted(
                        sensitivity[
                            "horizon"
                        ].unique()
                    )
                ],

            "state_bandwidths":
                [
                    float(x)
                    for x in sorted(
                        sensitivity[
                            "state_bandwidth"
                        ].unique()
                    )
                ],

            "temperatures":
                [
                    float(x)
                    for x in sorted(
                        sensitivity[
                            "temperature"
                        ].unique()
                    )
                ],

            "recency_decays":
                [
                    float(x)
                    for x in sorted(
                        sensitivity[
                            "recency_decay"
                        ].unique()
                    )
                ],

            "n_windows_grid":
                [
                    int(x)
                    for x in sorted(
                        sensitivity[
                            "n_windows"
                        ].unique()
                    )
                ],
        },


        "winner_stability":
            winner_summary,


        "temperature_effect":
            temperature_effect,


        "state_bandwidth_effect":
            bandwidth_effect,


        "recency_effect":
            recency_effect,


        "n_windows_effect":
            n_windows_effect,


        "top_candidate_configs":
            top_configs,


        "warnings":
            warnings,
    }


    return summary