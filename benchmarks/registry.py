"""
Dataset registry.

Available datasets have a loader in benchmarks/datasets.py.
Planned datasets are named here so ETTh1, Electricity, and
Traffic can be added without changing the evaluator.
This module does not load data and does not invent series.
"""


DATASET_REGISTRY = {
    "airpassengers": {
        "status": "available",
        "filename": "AirPassengers.csv",
        "target": "#Passengers",
        "frequency": "monthly",
        "known_seasonal_period": 12,
        "n_series": 1,
        "has_covariates": False,
        "description": (
            "Monthly international airline passenger counts."
        ),
    },
    "etth1": {
        "status": "planned",
        "filename": "ETTh1.csv",
        "target": "OT",
        "frequency": "hourly",
        "known_seasonal_period": 24,
        "n_series": 7,
        "has_covariates": False,
        "description": (
            "Electricity transformer temperature, hourly. "
            "Loader is not implemented."
        ),
    },
    "electricity": {
        "status": "planned",
        "filename": "electricity.csv",
        "target": None,
        "frequency": "hourly",
        "known_seasonal_period": 24,
        "n_series": None,
        "has_covariates": False,
        "description": (
            "Electricity consumption panel. "
            "Loader is not implemented."
        ),
    },
    "traffic": {
        "status": "planned",
        "filename": "traffic.csv",
        "target": None,
        "frequency": "hourly",
        "known_seasonal_period": 24,
        "n_series": None,
        "has_covariates": False,
        "description": (
            "Traffic occupancy panel. "
            "Loader is not implemented."
        ),
    },
}


def list_datasets():
    return list(DATASET_REGISTRY)


def get_dataset_spec(dataset_name):
    name = dataset_name.strip().lower()
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{dataset_name}' is not registered."
        )
    spec = dict(DATASET_REGISTRY[name])
    spec["name"] = name
    return spec
