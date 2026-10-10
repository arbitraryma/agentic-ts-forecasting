import importlib


MODEL_REGISTRY = {

    "seasonal_naive": {
        "family": "statistical",
        "module": "models.statistical.seasonal_naive",
        "class_name": "SeasonalNaive",
        "enabled": True,
    },

    "auto_arima": {
        "family": "statistical",
        "module": "models.statistical.arima",
        "class_name": "AutoArima",
        "enabled": True,
    },

    "ets": {
        "family": "statistical",
        "module": "models.statistical.ets",
        "class_name": "ETSModel",
        "enabled": True,
    },

    "xgboost": {
        "family": "ml",
        "module": "models.ml.xgboost_model",
        "class_name": "XGBoostModel",
        "enabled": True,
    },

    "dlinear": {
        "family": "deep",
        "module": "models.deep.dlinear",
        "class_name": "DLinearModel",
        "enabled": True,
    },

    "patchtst": {
        "family": "deep",
        "module": "models.deep.patchtst",
        "class_name": "PatchTSTModel",
        "enabled": True,
    },

    "timesnet": {
        "family": "deep",
        "module": "models.deep.timesnet",
        "class_name": "TimesNetModel",
        "enabled": True,
    },

    "itransformer": {
        "family": "deep",
        "module": "models.deep.itransformer",
        "class_name": "ITransformerModel",
        "enabled": True,
    },

    "chronos2": {
        "family": "foundation",
        "module": "models.foundation.chronos2",
        "class_name": "Chronos2Model",
        "enabled": True,
    },

    "timesfm3": {
        "family": "foundation",
        "module": "models.foundation.timesfm3",
        "class_name": "TimesFM3Model",
        "enabled": True,
    },

    "moirai2": {
        "family": "foundation",
        "module": "models.foundation.moirai2",
        "class_name": "Moirai2Model",
        "enabled": True,
    },
}


def get_model_class(model_name):

    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model: {model_name}"
        )

    config = MODEL_REGISTRY[
        model_name
    ]

    if not config["enabled"]:
        raise ValueError(
            f"Model '{model_name}' is disabled."
        )

    module = importlib.import_module(
        config["module"]
    )

    model_class = getattr(
        module,
        config["class_name"]
    )

    return model_class


def create_model(
    model_name,
    **kwargs,
):

    model_class = get_model_class(
        model_name
    )

    return model_class(
        **kwargs
    )


def list_enabled_models():

    return [
        model_name
        for model_name, config
        in MODEL_REGISTRY.items()
        if config["enabled"]
    ]