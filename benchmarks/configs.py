from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# Benchmark configurations
#
# YAML files in configs/ are the experiment source.
# These dictionaries remain so existing callers keep
# working when a YAML file is absent.
# =========================================================

# Dataset settings live in configs/*.yaml.
# This dictionary is only a fallback when default.yaml
# is missing. It does not name a dataset.
BENCHMARK_CONFIGS = {

    "default": {

        "horizon":
            3,

        "outer_windows":
            5,

        "inner_windows":
            5,

        "train_window":
            60,

        "window_types": [
            "expanding",
        ],

        "compute_budget":
            "medium",

        "compute_weight":
            0.10,

        "temperature":
            1.0,

        "fast_model_config":
            False,

        "model_allowlist":
            None,
    },
}


def load_yaml_config(path):
    """Load one YAML experiment file."""

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        loaded = yaml.safe_load(handle)

    if not isinstance(loaded, dict):
        raise ValueError(
            f"Config must be a mapping: {path}"
        )

    return loaded


def get_benchmark_config(
    dataset_name,
):
    """
    Retrieve benchmark configuration from configs/*.yaml.

    configs/default.yaml supplies the base settings.
    configs/<dataset>.yaml overrides that base.
    There is no dataset-specific setting in this module.
    """

    dataset_name = (
        dataset_name
        .strip()
        .lower()
    )

    default_path = (
        ROOT / "configs" / "default.yaml"
    )

    if default_path.exists():
        config = load_yaml_config(default_path)
    else:
        config = BENCHMARK_CONFIGS["default"].copy()

    specific = (
        ROOT / "configs" / f"{dataset_name}.yaml"
    )

    if (
        specific.exists()
        and specific.resolve() != default_path.resolve()
    ):
        config.update(
            load_yaml_config(specific)
        )

    return config


def resolve_experiment_config(
    config_path=None,
    dataset=None,
    overrides=None,
):
    """
    Build the config for one run.

    --config is merged on top of configs/default.yaml.
    --dataset keeps the previous CLI working.
    overrides come from explicit CLI flags.
    """

    if config_path is not None:
        config = load_yaml_config(
            ROOT / "configs" / "default.yaml"
        )
        config.update(
            load_yaml_config(config_path)
        )
        config["config_name"] = Path(
            config_path
        ).stem
    elif dataset is not None:
        config = get_benchmark_config(dataset)
        config["dataset"] = dataset.strip().lower()
        config["config_name"] = config["dataset"]
    else:
        raise ValueError(
            "Pass --config or --dataset."
        )

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value

    if not config.get("dataset"):
        raise ValueError(
            "The config must name a dataset."
        )

    config["dataset"] = (
        str(config["dataset"])
        .strip()
        .lower()
    )

    if config.get("model_allowlist") == "":
        config["model_allowlist"] = None

    return config