import numpy as np
import pandas as pd


def _softmax(
    values,
):
    """
    Numerically stable softmax.
    """

    values = np.asarray(
        values,
        dtype=float,
    )


    values = (
        values
        - np.max(values)
    )


    exp_values = np.exp(
        values
    )


    return (
        exp_values
        / exp_values.sum()
    )


def build_horizon_routing_table(
    all_results,
    horizon,
    compute_weight=0.10,
    temperature=0.50,
):
    """
    Build horizon-specific routing weights.

    For every expert:

        model + window_type

    compute a separate validation loss
    for each forecast horizon h.

    Then compute:

        utility_{m,h}

    and:

        weight_{m,h}

    Parameters
    ----------
    all_results : pd.DataFrame

        Must contain:

        model
        family
        window_type
        runtime
        abs_error_h1
        ...
        abs_error_hH

    horizon : int

        Forecast horizon.

    compute_weight : float

        Runtime penalty strength.

    temperature : float

        Softmax temperature.

    Returns
    -------
    pd.DataFrame

        Long-format table:

        model
        family
        window_type
        horizon
        horizon_mae
        runtime
        loss_regret
        compute_penalty
        utility
        routing_weight
    """

    if temperature <= 0:

        raise ValueError(
            "temperature must be positive."
        )


    if compute_weight < 0:

        raise ValueError(
            "compute_weight cannot be negative."
        )


    required_columns = [
        "model",
        "family",
        "window_type",
        "runtime",
    ]


    for column in required_columns:

        if column not in all_results.columns:

            raise ValueError(
                f"Missing required column: "
                f"{column}"
            )


    rows = []


    # ===================================
    # One horizon at a time
    # ===================================

    for h in range(
        1,
        horizon + 1,
    ):

        error_column = (
            f"abs_error_h{h}"
        )


        if error_column not in all_results.columns:

            raise ValueError(
                f"Missing horizon error "
                f"column: {error_column}"
            )


        # ===================================
        # Aggregate rolling windows
        # ===================================

        horizon_summary = (
            all_results

            .groupby(
                [
                    "family",
                    "model",
                    "window_type",
                ]
            )[
                [
                    error_column,
                    "runtime",
                ]
            ]

            .mean()

            .reset_index()
        )


        horizon_summary = (
            horizon_summary
            .rename(
                columns={
                    error_column:
                        "horizon_mae"
                }
            )
        )


        # ===================================
        # Best loss at this horizon
        # ===================================

        best_loss = float(
            horizon_summary[
                "horizon_mae"
            ].min()
        )


        best_loss = max(
            best_loss,
            1e-8,
        )


        horizon_summary[
            "loss_regret"
        ] = (
            (
                horizon_summary[
                    "horizon_mae"
                ]
                - best_loss
            )
            / best_loss
        )


        # ===================================
        # Runtime reference
        # ===================================

        positive_runtime = (
            horizon_summary.loc[
                horizon_summary[
                    "runtime"
                ] > 0,
                "runtime",
            ]
        )


        if len(
            positive_runtime
        ) == 0:

            runtime_reference = 1.0

        else:

            runtime_reference = float(
                positive_runtime.median()
            )


        runtime_reference = max(
            runtime_reference,
            1e-8,
        )


        # ===================================
        # Compute penalty
        # ===================================

        horizon_summary[
            "compute_penalty"
        ] = np.log1p(
            horizon_summary[
                "runtime"
            ]
            / runtime_reference
        )


        # ===================================
        # Utility_{m,h}
        # ===================================

        horizon_summary[
            "utility"
        ] = (
            -horizon_summary[
                "loss_regret"
            ]
            - compute_weight
            * horizon_summary[
                "compute_penalty"
            ]
        )


        # ===================================
        # Weight_{m,h}
        # ===================================

        logits = (
            horizon_summary[
                "utility"
            ].values
            / temperature
        )


        horizon_summary[
            "routing_weight"
        ] = _softmax(
            logits
        )


        horizon_summary[
            "horizon"
        ] = h


        # ===================================
        # Ranking
        # ===================================

        horizon_summary = (
            horizon_summary
            .sort_values(
                "utility",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        horizon_summary[
            "routing_rank"
        ] = (
            np.arange(
                len(
                    horizon_summary
                )
            )
            + 1
        )


        rows.append(
            horizon_summary
        )


    # ===================================
    # Combine horizons
    # ===================================

    routing_table = pd.concat(
        rows,
        ignore_index=True,
    )


    return routing_table


def get_horizon_weights(
    horizon_routing_table,
    model_names,
    window_types,
    horizon,
):
    """
    Retrieve normalized top-expert weights
    for one specific forecast horizon.

    Parameters
    ----------
    horizon_routing_table : pd.DataFrame

    model_names : list[str]

    window_types : list[str]

    horizon : int

    Returns
    -------
    np.ndarray
    """

    selected_weights = []


    for model_name, window_type in zip(
        model_names,
        window_types,
    ):

        match = horizon_routing_table[
            (
                horizon_routing_table[
                    "model"
                ]
                == model_name
            )
            &
            (
                horizon_routing_table[
                    "window_type"
                ]
                == window_type
            )
            &
            (
                horizon_routing_table[
                    "horizon"
                ]
                == horizon
            )
        ]


        if len(match) == 0:

            selected_weights.append(
                0.0
            )

        else:

            selected_weights.append(
                float(
                    match.iloc[
                        0
                    ][
                        "routing_weight"
                    ]
                )
            )


    selected_weights = np.asarray(
        selected_weights,
        dtype=float,
    )


    total = (
        selected_weights.sum()
    )


    if total <= 0:

        selected_weights = (
            np.ones(
                len(
                    selected_weights
                )
            )
            / len(
                selected_weights
            )
        )

    else:

        selected_weights = (
            selected_weights
            / total
        )


    return selected_weights