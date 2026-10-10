import numpy as np
import pandas as pd

from diagnostics.state import (
    extract_state_features,
)


STATE_FEATURES = [
    "trend_strength",
    "acf1",
    "recent_mean_shift",
    "recent_vol_ratio",
    "diff_vol_ratio",
    "seasonal_corr",
]


def _softmax(
    values,
):

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


def _standardize_states(
    historical_states,
    current_state,
):
    """
    Standardize state features so that
    variables on different scales are
    comparable in the distance calculation.
    """

    historical_states = np.asarray(
        historical_states,
        dtype=float,
    )


    current_state = np.asarray(
        current_state,
        dtype=float,
    )


    combined = np.vstack(
        [
            historical_states,
            current_state[
                None,
                :
            ],
        ]
    )


    mean = np.mean(
        combined,
        axis=0,
    )


    std = np.std(
        combined,
        axis=0,
    )


    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )


    historical_scaled = (
        historical_states
        - mean
    ) / std


    current_scaled = (
        current_state
        - mean
    ) / std


    return (
        historical_scaled,
        current_scaled,
    )


def _state_similarity_weights(
    historical_states,
    current_state,
    bandwidth=1.0,
):
    """
    Gaussian state-similarity kernel.

    More similar historical regimes receive
    larger weights.
    """

    if bandwidth <= 0:

        raise ValueError(
            "bandwidth must be positive."
        )


    (
        historical_scaled,
        current_scaled,
    ) = _standardize_states(
        historical_states,
        current_state,
    )


    distances_squared = np.sum(
        (
            historical_scaled
            - current_scaled
        ) ** 2,
        axis=1,
    )


    weights = np.exp(
        -0.5
        * distances_squared
        / (
            bandwidth ** 2
        )
    )


    return weights


def _recency_weights(
    windows,
    decay=0.0,
):
    """
    Exponential recency weighting.

    decay = 0:
        no recency preference.

    larger decay:
        stronger preference for
        recent forecast origins.
    """

    windows = np.asarray(
        windows,
        dtype=float,
    )


    if decay < 0:

        raise ValueError(
            "recency decay cannot be negative."
        )


    if decay == 0:

        return np.ones_like(
            windows
        )


    latest = np.max(
        windows
    )


    age = (
        latest
        - windows
    )


    return np.exp(
        -decay
        * age
    )


def _effective_sample_size(
    weights,
):
    """
    Effective sample size of localized weights.
    """

    weights = np.asarray(
        weights,
        dtype=float,
    )


    total = np.sum(
        weights
    )


    if total <= 0:

        return 0.0


    normalized = (
        weights
        / total
    )


    denominator = np.sum(
        normalized ** 2
    )


    if denominator <= 0:

        return 0.0


    return float(
        1.0
        / denominator
    )


def build_state_horizon_routing_table(
    all_results,
    current_y,
    horizon,
    seasonal_period=None,
    compute_weight=0.10,
    temperature=0.50,
    state_bandwidth=1.0,
    recency_decay=0.0,
):
    """
    State-aware + horizon-specific routing.

    Produces:

        w_{m,h,t}

    instead of a single model weight w_m.

    Validation windows similar to the current
    time-series state receive larger influence.

    Parameters
    ----------
    all_results : pd.DataFrame
        Window-level rolling CV results.

    current_y : array-like
        Current observed time series.

    horizon : int
        Forecast horizon.

    seasonal_period : int or None

    compute_weight : float
        Runtime penalty.

    temperature : float
        Softmax temperature.

    state_bandwidth : float
        Bandwidth of state similarity kernel.

    recency_decay : float
        Optional recency weighting.

    Returns
    -------
    pd.DataFrame
    """

    if temperature <= 0:

        raise ValueError(
            "temperature must be positive."
        )


    # ===================================
    # Current regime
    # ===================================

    current_state_dict = (
        extract_state_features(
            current_y,
            seasonal_period=
                seasonal_period,
        )
    )


    current_state = np.asarray(
        [
            current_state_dict[
                feature
            ]
            for feature in STATE_FEATURES
        ],
        dtype=float,
    )


    # ===================================
    # Validate state columns
    # ===================================

    state_columns = [
        f"state_{feature}"
        for feature
        in STATE_FEATURES
    ]


    for column in state_columns:

        if column not in all_results.columns:

            raise ValueError(
                f"Missing state column: "
                f"{column}"
            )


    rows = []


    expert_groups = (
        all_results.groupby(
            [
                "family",
                "model",
                "window_type",
            ],
            sort=False,
        )
    )


    # ===================================
    # Horizon-specific routing
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
                f"Missing column "
                f"{error_column}."
            )


        horizon_rows = []


        # ===================================
        # Estimate localized loss
        # for each expert
        # ===================================

        for (
            family,
            model,
            window_type,
        ), group in expert_groups:

            group = (
                group
                .sort_values(
                    "window"
                )
                .reset_index(
                    drop=True
                )
            )


            historical_states = (
                group[
                    state_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


            state_weights = (
                _state_similarity_weights(
                    historical_states=
                        historical_states,

                    current_state=
                        current_state,

                    bandwidth=
                        state_bandwidth,
                )
            )


            recency_weights = (
                _recency_weights(
                    windows=
                        group[
                            "window"
                        ].values,

                    decay=
                        recency_decay,
                )
            )


            local_weights = (
                state_weights
                * recency_weights
            )


            weight_sum = float(
                np.sum(
                    local_weights
                )
            )


            # Fallback if similarities
            # numerically vanish.
            if weight_sum <= 1e-12:

                local_weights = (
                    np.ones(
                        len(group),
                        dtype=float,
                    )
                )

                weight_sum = float(
                    len(group)
                )


            # ===================================
            # Localized horizon loss
            # ===================================

            errors = (
                group[
                    error_column
                ]
                .to_numpy(
                    dtype=float
                )
            )


            localized_loss = float(
                np.sum(
                    local_weights
                    * errors
                )
                / weight_sum
            )


            # ===================================
            # Localized runtime
            # ===================================

            runtimes = (
                group[
                    "runtime"
                ]
                .to_numpy(
                    dtype=float
                )
            )


            localized_runtime = float(
                np.sum(
                    local_weights
                    * runtimes
                )
                / weight_sum
            )


            ess = (
                _effective_sample_size(
                    local_weights
                )
            )


            horizon_rows.append({

                "family":
                    family,

                "model":
                    model,

                "window_type":
                    window_type,

                "horizon":
                    h,

                "localized_loss":
                    localized_loss,

                "runtime":
                    localized_runtime,

                "state_weight_sum":
                    weight_sum,

                "effective_sample_size":
                    ess,
            })


        horizon_table = pd.DataFrame(
            horizon_rows
        )


        # ===================================
        # Relative validation regret
        # ===================================

        best_loss = float(
            horizon_table[
                "localized_loss"
            ].min()
        )


        best_loss = max(
            best_loss,
            1e-8,
        )


        horizon_table[
            "loss_regret"
        ] = (
            (
                horizon_table[
                    "localized_loss"
                ]
                - best_loss
            )
            / best_loss
        )


        # ===================================
        # Runtime penalty
        # ===================================

        positive_runtime = (
            horizon_table.loc[
                horizon_table[
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


        horizon_table[
            "compute_penalty"
        ] = np.log1p(
            horizon_table[
                "runtime"
            ]
            / runtime_reference
        )


        # ===================================
        # State-aware utility
        # ===================================

        horizon_table[
            "utility"
        ] = (
            -horizon_table[
                "loss_regret"
            ]
            - compute_weight
            * horizon_table[
                "compute_penalty"
            ]
        )


        # ===================================
        # w_{m,h,t}
        # ===================================

        logits = (
            horizon_table[
                "utility"
            ].values
            / temperature
        )


        horizon_table[
            "routing_weight"
        ] = _softmax(
            logits
        )


        horizon_table = (
            horizon_table
            .sort_values(
                "utility",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )


        horizon_table[
            "routing_rank"
        ] = (
            np.arange(
                len(
                    horizon_table
                )
            )
            + 1
        )


        rows.append(
            horizon_table
        )


    return pd.concat(
        rows,
        ignore_index=True,
    )