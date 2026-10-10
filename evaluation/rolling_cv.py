import time

import numpy as np
import pandas as pd

from diagnostics.state import (
    extract_state_features,
)


def _mae(
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
                y_true
                - y_pred
            )
        )
    )


def _rmse(
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
                    y_true
                    - y_pred
                ) ** 2
            )
        )
    )


def rolling_backtest(
    model_class,
    y,
    horizon,
    n_windows,
    model_kwargs=None,
    window_type="expanding",
    train_window=None,
    seasonal_period=None,
):
    """
    Rolling-origin forecasting evaluation.

    Records:

    - MAE
    - RMSE
    - runtime
    - horizon-specific errors
    - state/regime features

    These outputs support:

        model routing
        horizon-specific routing
        state-aware routing
    """

    if model_kwargs is None:
        model_kwargs = {}


    # ===================================
    # Clean time series
    # ===================================

    y = (
        pd.Series(y)
        .dropna()
        .astype(float)
        .reset_index(drop=True)
    )


    # ===================================
    # Validation
    # ===================================

    if horizon <= 0:

        raise ValueError(
            "horizon must be positive."
        )


    if n_windows <= 0:

        raise ValueError(
            "n_windows must be positive."
        )


    if window_type not in {
        "expanding",
        "sliding",
    }:

        raise ValueError(
            "window_type must be "
            "'expanding' or 'sliding'."
        )


    if (
        window_type
        == "sliding"
        and train_window is None
    ):

        raise ValueError(
            "train_window must be provided "
            "for sliding-window backtesting."
        )


    if (
        window_type
        == "sliding"
        and train_window <= 0
    ):

        raise ValueError(
            "train_window must be positive."
        )


    total_test_size = (
        horizon
        * n_windows
    )


    initial_train_size = (
        len(y)
        - total_test_size
    )


    if initial_train_size <= 0:

        raise ValueError(
            "Not enough observations "
            "for requested rolling CV."
        )


    results = []


    # ===================================
    # Rolling origins
    # ===================================

    for window_idx in range(
        n_windows
    ):

        train_end = (
            initial_train_size
            + window_idx
            * horizon
        )


        test_start = train_end


        test_end = (
            test_start
            + horizon
        )


        # ===================================
        # Training-window strategy
        # ===================================

        if window_type == "expanding":

            train_start = 0

        else:

            train_start = max(
                0,
                train_end
                - train_window,
            )


        train = y.iloc[
            train_start:train_end
        ]


        test = y.iloc[
            test_start:test_end
        ]


        if len(test) != horizon:

            continue


        # ===================================
        # State at this forecast origin
        # ===================================

        state = (
            extract_state_features(
                train,
                seasonal_period=
                    seasonal_period,
            )
        )


        # ===================================
        # Instantiate model
        # ===================================

        model = model_class(
            **model_kwargs
        )


        # ===================================
        # Fit runtime
        # ===================================

        fit_start = (
            time.perf_counter()
        )


        model.fit(
            train
        )


        fit_runtime = (
            time.perf_counter()
            - fit_start
        )


        # ===================================
        # Prediction runtime
        # ===================================

        predict_start = (
            time.perf_counter()
        )


        forecast = model.predict(
            horizon=horizon
        )


        predict_runtime = (
            time.perf_counter()
            - predict_start
        )


        total_runtime = (
            fit_runtime
            + predict_runtime
        )


        # ===================================
        # Forecast formatting
        # ===================================

        forecast = np.asarray(
            forecast,
            dtype=float,
        )


        forecast = np.squeeze(
            forecast
        )


        if forecast.ndim != 1:

            raise ValueError(
                f"Expected 1D forecast "
                f"but got shape "
                f"{forecast.shape}."
            )


        if len(
            forecast
        ) != horizon:

            raise ValueError(
                f"Forecast length "
                f"{len(forecast)} "
                f"does not match "
                f"horizon {horizon}."
            )


        y_true = (
            test
            .to_numpy(
                dtype=float
            )
        )


        # ===================================
        # Overall errors
        # ===================================

        mae_value = _mae(
            y_true,
            forecast,
        )


        rmse_value = _rmse(
            y_true,
            forecast,
        )


        # ===================================
        # Base result
        # ===================================

        result = {

            "window":
                window_idx + 1,

            "window_type":
                window_type,

            "train_start":
                train_start,

            "train_end":
                train_end,

            "test_start":
                test_start,

            "test_end":
                test_end,

            "train_size":
                len(train),

            "test_size":
                len(test),

            "mae":
                mae_value,

            "rmse":
                rmse_value,

            "fit_runtime":
                fit_runtime,

            "predict_runtime":
                predict_runtime,

            "runtime":
                total_runtime,
        }


        # ===================================
        # Add state features
        # ===================================

        for (
            state_name,
            state_value,
        ) in state.items():

            result[
                f"state_{state_name}"
            ] = float(
                state_value
            )


        # ===================================
        # Horizon-specific errors
        # ===================================

        absolute_errors = (
            np.abs(
                y_true
                - forecast
            )
        )


        squared_errors = (
            (
                y_true
                - forecast
            ) ** 2
        )


        for h in range(
            horizon
        ):

            horizon_id = (
                h + 1
            )


            result[
                f"abs_error_h{horizon_id}"
            ] = float(
                absolute_errors[h]
            )


            result[
                f"sq_error_h{horizon_id}"
            ] = float(
                squared_errors[h]
            )


        results.append(
            result
        )


    return pd.DataFrame(
        results
    )