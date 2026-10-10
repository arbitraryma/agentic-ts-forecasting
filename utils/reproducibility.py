"""
Reproducibility helpers.

The seed comes from the experiment config.
Setting it does not by itself make GPU or MPS
kernels bit-identical.
"""

import logging
import os
import random
import warnings


def configure_runtime(verbose=False):
    """
    Set thread limits before numerical libraries fan out.

    Quiet mode reduces Lightning console noise.
    Model wrappers still own their Trainer objects,
    so a Lightning summary table can still appear.
    """

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if verbose:
        os.environ.pop("TQDM_DISABLE", None)
        level = logging.INFO
    else:
        os.environ["TQDM_DISABLE"] = "1"
        level = logging.ERROR
        warnings.filterwarnings(
            "ignore",
            module="lightning",
        )
        warnings.filterwarnings(
            "ignore",
            module="pytorch_lightning",
        )
        warnings.filterwarnings(
            "ignore",
            module="neuralforecast",
        )

    for logger_name in (
        "lightning",
        "pytorch_lightning",
        "lightning.pytorch",
        "lightning.fabric",
    ):
        logging.getLogger(logger_name).setLevel(level)


def set_global_seed(seed):
    """Seed Python, NumPy, and PyTorch from one config seed."""

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if (
        hasattr(torch, "mps")
        and hasattr(torch.mps, "manual_seed")
        and torch.backends.mps.is_available()
    ):
        torch.mps.manual_seed(seed)

    return seed


def get_logger(name):
    """Console logger with a stable one-line format."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(message)s")
        )
        logger.addHandler(handler)

    return logger
