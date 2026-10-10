from models.registry import MODEL_REGISTRY


def route_candidates(
    diagnostics,
    horizon,
    n_series=1,
    has_covariates=False,
    compute_budget="medium",
):
    """
    Select a reasonable subset of forecasting models
    before rolling validation.

    Parameters
    ----------
    diagnostics : dict
        Output from basic_diagnostics().

    horizon : int
        Forecast horizon.

    n_series : int
        Number of target series / variables.

    has_covariates : bool
        Whether useful external covariates are available.

    compute_budget : str
        "low", "medium", or "high".

    Returns
    -------
    candidates : list[str]
        Selected model names.

    reasons : dict[str, list[str]]
        Human-readable routing reasons.
    """

    n_obs = diagnostics.get("n_obs", 0)
    seasonal_period = diagnostics.get("seasonal_period")
    trend_slope = diagnostics.get("trend_slope", 0.0)
    missing_rate = diagnostics.get("missing_rate", 0.0)

    candidates = []
    reasons = {}


    def add_model(model_name, reason):

        if model_name not in MODEL_REGISTRY:
            return

        if not MODEL_REGISTRY[model_name]["enabled"]:
            return

        if model_name not in candidates:
            candidates.append(model_name)
            reasons[model_name] = []

        reasons[model_name].append(reason)


    # ===================================
    # 1. Always keep strong/simple baselines
    # ===================================

    add_model(
        "auto_arima",
        "Strong statistical baseline for autoregressive structure.",
    )

    add_model(
        "ets",
        "Strong statistical baseline for trend and smooth temporal structure.",
    )


    # ===================================
    # 2. Seasonality-aware models
    # ===================================

    if seasonal_period is not None:

        add_model(
            "seasonal_naive",
            f"Detected seasonality with period {seasonal_period}.",
        )

        add_model(
            "auto_arima",
            "Detected seasonality supports seasonal ARIMA structure.",
        )

        add_model(
            "ets",
            "Detected seasonality supports seasonal exponential smoothing.",
        )


    # ===================================
    # 3. Machine-learning model
    # ===================================

    if n_obs >= 40:

        add_model(
            "xgboost",
            "Series is long enough to construct lagged supervised features.",
        )


    # ===================================
    # 4. Deep models
    # ===================================

    if n_obs >= 80:

        add_model(
            "dlinear",
            "Sufficient observations for a lightweight deep forecasting model.",
        )

    if n_obs >= 100:

        add_model(
            "patchtst",
            "Sequence is sufficiently long for patch-based Transformer learning.",
        )

        add_model(
            "timesnet",
            "Sequence is sufficiently long for learned multi-period temporal patterns.",
        )


    # ===================================
    # 5. Multivariate-specific model
    # ===================================

    if n_series > 1 and n_obs >= 80:

        add_model(
            "itransformer",
            "Multivariate target allows iTransformer to model cross-variable dependencies.",
        )


    # ===================================
    # 6. Foundation models
    # ===================================

    if compute_budget in {
        "medium",
        "high",
    }:

        add_model(
            "chronos2",
            "Pretrained foundation model provides a strong zero-shot candidate.",
        )

        add_model(
            "timesfm3",
            "Pretrained foundation model provides a strong zero-shot candidate.",
        )


    if compute_budget == "high":

        add_model(
            "moirai2",
            "High compute budget allows an additional pretrained foundation model.",
        )


    # ===================================
    # 7. Covariate-aware preference
    # ===================================

    if has_covariates:

        add_model(
            "chronos2",
            "Covariates are available and the wrapper supports covariate-aware forecasting.",
        )

        add_model(
            "timesfm3",
            "Covariates are available and the wrapper supports covariate-aware forecasting.",
        )


    # ===================================
    # 8. Missing-data safeguard
    # ===================================

    if missing_rate > 0.10:

        # This does not remove models yet.
        # We simply record the condition for later preprocessing.
        for model_name in candidates:

            reasons[
                model_name
            ].append(
                "Missing rate exceeds 10%; preprocessing or imputation may be required."
            )


    # ===================================
    # 9. Horizon information
    # ===================================

    if horizon >= 24:

        for model_name in [
            "chronos2",
            "timesfm3",
            "patchtst",
        ]:

            if model_name in candidates:

                reasons[
                    model_name
                ].append(
                    "Longer forecast horizon makes this model worth validating."
                )


    return candidates, reasons