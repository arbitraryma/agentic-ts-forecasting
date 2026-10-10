import numpy as np
import pandas as pd


def _safe_corr(x, y):
    """
    Safe correlation helper.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    if len(x) < 2:
        return 0.0

    if np.std(x) < 1e-8:
        return 0.0

    if np.std(y) < 1e-8:
        return 0.0

    corr = np.corrcoef(
        x,
        y,
    )[0, 1]

    if not np.isfinite(corr):
        return 0.0

    return float(corr)


def extract_state_features(
    y,
    seasonal_period=None,
    recent_fraction=0.25,
):
    """
    Extract a compact state vector describing
    the current time-series regime.

    Features
    --------
    trend_strength
        Normalized trend over the observed window.

    acf1
        Lag-1 dependence.

    recent_mean_shift
        Recent mean relative to historical mean.

    recent_vol_ratio
        Recent volatility relative to historical volatility.

    diff_vol_ratio
        Volatility of first differences relative
        to level volatility.

    seasonal_corr
        Correlation at the detected seasonal lag.

    Parameters
    ----------
    y : array-like
        Univariate time series.

    seasonal_period : int or None
        Optional detected seasonal period.

    recent_fraction : float
        Fraction of the latest observations
        defining the recent regime.

    Returns
    -------
    dict
    """

    y = (
        pd.Series(y)
        .dropna()
        .astype(float)
        .reset_index(drop=True)
    )

    n = len(y)

    if n < 4:
        raise ValueError(
            "At least 4 observations are required "
            "to extract state features."
        )


    values = y.to_numpy(
        dtype=float
    )


    # ===================================
    # Basic scale
    # ===================================

    overall_mean = float(
        np.mean(values)
    )

    overall_std = float(
        np.std(
            values,
            ddof=0,
        )
    )

    scale = max(
        overall_std,
        1e-8,
    )


    # ===================================
    # Normalized trend
    # ===================================

    x = np.arange(
        n,
        dtype=float,
    )

    slope = float(
        np.polyfit(
            x,
            values,
            1,
        )[0]
    )


    # Scale trend so series with different
    # magnitudes become comparable.
    trend_strength = (
        slope
        * n
        / scale
    )


    # ===================================
    # Lag-1 dependence
    # ===================================

    acf1 = _safe_corr(
        values[:-1],
        values[1:],
    )


    # ===================================
    # Recent regime
    # ===================================

    recent_length = max(
        3,
        int(
            np.ceil(
                n
                * recent_fraction
            )
        ),
    )

    recent_length = min(
        recent_length,
        n,
    )


    recent = values[
        -recent_length:
    ]


    recent_mean = float(
        np.mean(
            recent
        )
    )


    recent_std = float(
        np.std(
            recent,
            ddof=0,
        )
    )


    recent_mean_shift = (
        recent_mean
        - overall_mean
    ) / scale


    recent_vol_ratio = (
        recent_std
        / scale
    )


    # ===================================
    # Difference volatility
    # ===================================

    differences = np.diff(
        values
    )


    diff_std = float(
        np.std(
            differences,
            ddof=0,
        )
    )


    diff_vol_ratio = (
        diff_std
        / scale
    )


    # ===================================
    # Seasonal dependence
    # ===================================

    seasonal_corr = 0.0


    if (
        seasonal_period is not None
        and seasonal_period >= 2
        and n > seasonal_period
    ):

        seasonal_corr = _safe_corr(
            values[
                :-seasonal_period
            ],
            values[
                seasonal_period:
            ],
        )


    # ===================================
    # State vector
    # ===================================

    return {

        "trend_strength":
            float(
                trend_strength
            ),

        "acf1":
            float(
                acf1
            ),

        "recent_mean_shift":
            float(
                recent_mean_shift
            ),

        "recent_vol_ratio":
            float(
                recent_vol_ratio
            ),

        "diff_vol_ratio":
            float(
                diff_vol_ratio
            ),

        "seasonal_corr":
            float(
                seasonal_corr
            ),
    }