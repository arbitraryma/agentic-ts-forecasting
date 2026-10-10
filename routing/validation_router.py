import numpy as np
import pandas as pd


def _softmax(values):
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


def compute_routing_scores(
    validation_summary,
    metric="mae",
    compute_weight=0.10,
    temperature=0.50,
):
    """
    Compute validation-driven routing scores.

    Utility combines:

    1. Relative validation regret
    2. Log-scaled compute penalty

    Higher utility = better expert.

    Parameters
    ----------
    validation_summary : pd.DataFrame

        Expected columns:
        - family
        - model
        - window_type
        - mae
        - rmse
        - runtime

    metric : str
        Validation metric used for routing.

    compute_weight : float
        Strength of runtime penalty.

    temperature : float
        Softmax temperature.

    Returns
    -------
    pd.DataFrame
    """

    if not isinstance(
        validation_summary,
        pd.DataFrame,
    ):
        raise TypeError(
            "validation_summary must be "
            "a pandas DataFrame."
        )


    if validation_summary.empty:

        raise ValueError(
            "validation_summary is empty."
        )


    if metric not in validation_summary.columns:

        raise ValueError(
            f"Metric '{metric}' not found."
        )


    if temperature <= 0:

        raise ValueError(
            "temperature must be positive."
        )


    if compute_weight < 0:

        raise ValueError(
            "compute_weight cannot be negative."
        )


    df = validation_summary.copy()


    # ===================================
    # Runtime safeguard
    # ===================================

    if "runtime" not in df.columns:

        df[
            "runtime"
        ] = 0.0


    # ===================================
    # Clean invalid values
    # ===================================

    df = (
        df
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                metric,
                "runtime",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if df.empty:

        raise ValueError(
            "No valid routing rows remain."
        )


    # ===================================
    # Validation regret
    #
    # best model -> 0
    #
    # Example:
    #
    # best MAE = 10
    # model MAE = 12
    #
    # regret = 0.20
    # ===================================

    best_loss = float(
        df[
            metric
        ].min()
    )


    if best_loss <= 0:

        best_loss = 1e-8


    df[
        "loss_regret"
    ] = (
        (
            df[
                metric
            ]
            - best_loss
        )
        / best_loss
    )


    # ===================================
    # Robust compute reference
    # ===================================

    positive_runtime = (
        df.loc[
            df["runtime"] > 0,
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
    # Log compute penalty
    #
    # Prevent very large runtimes
    # from dominating everything.
    # ===================================

    df[
        "compute_penalty"
    ] = np.log1p(
        df[
            "runtime"
        ]
        / runtime_reference
    )


    # ===================================
    # Utility
    #
    # higher = better
    # ===================================

    df[
        "utility"
    ] = (
        -df[
            "loss_regret"
        ]
        - compute_weight
        * df[
            "compute_penalty"
        ]
    )


    # ===================================
    # Soft routing weights
    # ===================================

    logits = (
        df[
            "utility"
        ].values
        / temperature
    )


    df[
        "routing_weight"
    ] = _softmax(
        logits
    )


    # ===================================
    # Ranking
    # ===================================

    df = (
        df
        .sort_values(
            by="utility",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    df[
        "routing_rank"
    ] = (
        np.arange(
            len(df)
        )
        + 1
    )


    return df


def select_top_models(
    routing_table,
    top_k=3,
):
    """
    Select top-k experts according to utility.
    """

    if top_k <= 0:

        raise ValueError(
            "top_k must be positive."
        )


    return (
        routing_table
        .sort_values(
            by="utility",
            ascending=False,
        )
        .head(
            top_k
        )
        .reset_index(
            drop=True
        )
    )