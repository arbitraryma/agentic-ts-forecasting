from pathlib import Path

import pandas as pd

from benchmarks.registry import (
    get_dataset_spec,
)


# =========================================================
# Unified dataset loader
# =========================================================

def load_dataset(
    dataset_name,
    data_root="data",
    target=None,
):
    """
    Load a supported forecasting dataset.

    Returns
    -------
    dict
        {
            "name": str,
            "y": pd.Series,
            "n_series": int,
            "has_covariates": bool,
            "frequency": str or None,
            "known_seasonal_period": int or None,
            "metadata": dict,
        }

    Notes
    -----
    For now, this benchmark runner focuses on
    univariate forecasting.

    Multivariate / panel support can be added later.
    """

    dataset_name = (
        dataset_name
        .strip()
        .lower()
    )

    if dataset_name in {
        "air_passengers",
        "air",
    }:
        dataset_name = "airpassengers"

    data_root = Path(
        data_root
    )

    try:
        spec = get_dataset_spec(dataset_name)
    except KeyError:
        spec = None

    if spec is not None and spec["status"] == "planned":
        raise ValueError(
            f"Dataset '{dataset_name}' is registered "
            "but its loader is not implemented. "
            f"Expected future file: "
            f"{data_root / spec['filename']}."
        )


    # =====================================================
    # AirPassengers
    # =====================================================

    if dataset_name in {
        "airpassengers",
        "air_passengers",
        "air",
    }:

        path = (
            data_root
            / "AirPassengers.csv"
        )


        if not path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )


        df = pd.read_csv(
            path
        )


        column = target or "#Passengers"

        if column not in df.columns:

            raise ValueError(
                "AirPassengers.csv must contain "
                f"column '{column}'."
            )


        y = (
            df[
                column
            ]
            .astype(float)
            .reset_index(
                drop=True
            )
        )


        return _dataset_record(
            name="airpassengers",
            y=y,
            n_series=1,
            has_covariates=False,
            frequency="monthly",
            known_seasonal_period=12,
            description=(
                "Monthly international airline "
                "passenger counts."
            ),
            target=column,
        )


    # =====================================================
    # Generic CSV
    #
    # Expected:
    #
    # data/<dataset_name>.csv
    #
    # Pass target to name the numeric column.
    # When target is omitted, the column name is "y".
    #
    # This is useful for quickly adding new datasets
    # without changing Python code.
    # =====================================================

    generic_path = (
        data_root
        / f"{dataset_name}.csv"
    )


    if generic_path.exists():

        df = pd.read_csv(
            generic_path
        )


        column = target or "y"

        if column not in df.columns:

            raise ValueError(
                f"{generic_path} exists, but must "
                f"contain a numeric column named '{column}'."
            )


        y = (
            df[
                column
            ]
            .astype(float)
            .reset_index(
                drop=True
            )
        )


        return _dataset_record(
            name=dataset_name,
            y=y,
            n_series=1,
            has_covariates=False,
            frequency=None,
            known_seasonal_period=None,
            description="Generic univariate CSV dataset.",
            target=column,
        )


    # =====================================================
    # Unknown dataset
    # =====================================================

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. "
        f"Expected a registered dataset or "
        f"{generic_path} with a numeric target column."
    )


def _dataset_record(
    name,
    y,
    n_series,
    has_covariates,
    frequency,
    known_seasonal_period,
    description,
    target,
):
    """Return the loader contract used by the evaluator."""

    return {
        "name": name,
        "y": y,
        "n_series": n_series,
        "has_covariates": has_covariates,
        "frequency": frequency,
        "known_seasonal_period": known_seasonal_period,
        "metadata": {
            "frequency": frequency,
            "known_seasonal_period": known_seasonal_period,
            "description": description,
            "target": target,
        },
    }