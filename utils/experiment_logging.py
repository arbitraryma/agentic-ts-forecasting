"""
Experiment folders.

Each run is stored under:

    results/<experiment_name>/<timestamp>_<config_hash>/

An existing directory is never replaced.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml


def config_hash(config):
    payload = json.dumps(
        config,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:8]


def create_experiment_dir(
    experiment_name,
    config,
    results_root="results",
):
    """
    Create a new directory for one run.

    The folder name is a timestamp plus a hash of the
    resolved config, including CLI overrides.
    """

    root = Path(results_root) / experiment_name
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = config_hash(config)
    directory = root / f"{stamp}_{digest}"

    suffix = 2
    while directory.exists():
        directory = root / f"{stamp}_{digest}_{suffix}"
        suffix += 1

    directory.mkdir(parents=False, exist_ok=False)
    return directory


def save_config(directory, config):
    path = Path(directory) / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config,
            handle,
            sort_keys=False,
        )
    return path


def save_frame(directory, name, frame):
    path = Path(directory) / name
    frame.to_csv(path, index=False)
    return path


def save_text_log(directory, lines):
    path = Path(directory) / "run.log"
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return path
