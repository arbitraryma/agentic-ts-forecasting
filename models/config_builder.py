def build_model_kwargs(
    model_name,
    diagnostics,
    fast=False,
    horizon=12,
    n_series=1,
):

    seasonal_period = diagnostics.get(
        "seasonal_period"
    )

    n_obs = diagnostics.get(
        "n_obs"
    )


    # ===================================
    # Shared data-dependent settings
    # ===================================

    input_length = _choose_input_length(
        n_obs=n_obs,
        seasonal_period=seasonal_period,
        horizon=horizon,
    )

    lag_length = _choose_lag_length(
        n_obs=n_obs,
        seasonal_period=seasonal_period,
        horizon=horizon,
    )


    # ===================================
    # Statistical models
    # ===================================

    if model_name == "seasonal_naive":

        kwargs = {
            "season_length":
                seasonal_period or 1
        }


    elif model_name == "auto_arima":

        kwargs = {
            "seasonal":
                seasonal_period is not None,

            "m":
                seasonal_period or 1,
        }


    elif model_name == "ets":

        kwargs = {
            "seasonal_periods":
                seasonal_period,

            "trend":
                "add",

            "seasonal":
                "add",
        }


    # ===================================
    # Machine learning
    # ===================================

    elif model_name == "xgboost":

        kwargs = {
            "lags":
                lag_length,

            "n_estimators":
                200,

            "max_depth":
                3,

            "learning_rate":
                0.05,
        }


    # ===================================
    # Deep learning
    # ===================================

    elif model_name == "dlinear":

        kwargs = {
            "input_length":
                input_length,

            "horizon":
                horizon,

            "moving_avg":
                _choose_moving_average(
                    input_length
                ),

            "epochs":
                50,

            "batch_size":
                32,
        }


    elif model_name == "patchtst":

        patch_length = min(
            8,
            max(
                2,
                input_length // 4
            )
        )

        kwargs = {
            "input_length":
                input_length,

            "horizon":
                horizon,

            "patch_length":
                patch_length,

            "patch_stride":
                max(
                    1,
                    patch_length // 2
                ),

            "d_model":
                64,

            "num_hidden_layers":
                2,

            "num_attention_heads":
                4,

            "epochs":
                20,

            "batch_size":
                32,
        }


    elif model_name == "timesnet":

        kwargs = {
            "input_length":
                input_length,

            "horizon":
                horizon,

            "top_k":
                3,

            "hidden_size":
                32,

            "encoder_layers":
                2,

            "max_steps":
                100,
        }


    elif model_name == "itransformer":

        kwargs = {
            "input_length":
                input_length,

            "horizon":
                horizon,

            "n_series":
                n_series,

            "hidden_size":
                64,

            "n_heads":
                4,

            "e_layers":
                2,

            "max_steps":
                100,
        }


    # ===================================
    # Foundation models
    # ===================================

    elif model_name == "chronos2":

        kwargs = {
            "device":
                "cpu",
        }


    elif model_name == "timesfm3":

        kwargs = {
            "device":
                "cpu",
        }


    elif model_name == "moirai2":

        kwargs = {
            "context_length":
                input_length,
        }


    else:

        kwargs = {}


    # ===================================
    # Fast-mode overrides
    # ===================================

    if fast:

        kwargs.update(
            _fast_overrides(
                model_name=model_name,
                horizon=horizon,
                n_series=n_series,
            )
        )


    return kwargs


# ===================================
# Input-length helper
# ===================================

def _choose_input_length(
    n_obs,
    seasonal_period,
    horizon,
):

    if seasonal_period is not None:

        desired = max(
            2 * seasonal_period,
            2 * horizon,
        )

    else:

        desired = max(
            2 * horizon,
            16,
        )


    if n_obs is None:

        return desired


    # Do not consume almost the entire
    # training series as context.
    max_allowed = max(
        8,
        n_obs // 2,
    )

    return max(
        8,
        min(
            desired,
            max_allowed,
        )
    )


# ===================================
# Lag helper
# ===================================

def _choose_lag_length(
    n_obs,
    seasonal_period,
    horizon,
):

    if seasonal_period is not None:

        desired = seasonal_period

    else:

        desired = max(
            horizon,
            8,
        )


    if n_obs is None:

        return desired


    return max(
        2,
        min(
            desired,
            max(
                2,
                n_obs // 3,
            )
        )
    )


# ===================================
# Moving-average helper
# ===================================

def _choose_moving_average(
    input_length
):

    moving_avg = min(
        25,
        input_length,
    )

    # Prefer odd kernel size
    if moving_avg % 2 == 0:
        moving_avg -= 1

    return max(
        3,
        moving_avg,
    )


# ===================================
# Fast-mode overrides
# ===================================

def _fast_overrides(
    model_name,
    horizon,
    n_series,
):

    if model_name == "xgboost":

        return {
            "n_estimators":
                20,

            "max_depth":
                2,
        }


    if model_name == "dlinear":

        return {
            "input_length":
                8,

            "horizon":
                horizon,

            "moving_avg":
                3,

            "epochs":
                1,

            "batch_size":
                32,
        }


    if model_name == "patchtst":

        return {
            "input_length":
                8,

            "horizon":
                horizon,

            "patch_length":
                4,

            "patch_stride":
                4,

            "d_model":
                16,

            "num_hidden_layers":
                1,

            "num_attention_heads":
                2,

            "epochs":
                1,

            "batch_size":
                32,
        }


    if model_name == "timesnet":

        return {
            "input_length":
                8,

            "horizon":
                horizon,

            "top_k":
                1,

            "hidden_size":
                8,

            "encoder_layers":
                1,

            "max_steps":
                1,
        }


    if model_name == "itransformer":

        return {
            "input_length":
                8,

            "horizon":
                horizon,

            "n_series":
                n_series,

            "hidden_size":
                8,

            "n_heads":
                2,

            "e_layers":
                1,

            "max_steps":
                1,
        }


    if model_name == "chronos2":

        return {
            "device":
                "cpu",
        }


    if model_name == "timesfm3":

        return {
            "device":
                "cpu",
        }


    if model_name == "moirai2":

        return {
            "context_length":
                16,
        }


    return {}