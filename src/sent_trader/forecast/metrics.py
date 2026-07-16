"""Evaluation metrics for quantile forecasts.

Pinball loss is the proper scoring rule for quantiles: being well-calibrated
AND sharp is the only way to minimize it. Coverage checks honesty directly:
a 90% interval that contains reality 70% of the time is lying about risk.
"""

import numpy as np
import pandas as pd

from . import TAUS


def pinball_loss(y_true: np.ndarray, quantile_preds: pd.DataFrame | np.ndarray) -> float:
    """Mean pinball loss across all quantiles (lower is better)."""
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    q = np.asarray(quantile_preds, dtype=float)
    taus = np.asarray(TAUS).reshape(1, -1)
    diff = y - q
    return float(np.mean(np.maximum(taus * diff, (taus - 1) * diff)))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of realized values inside [lower, upper]."""
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def report(name: str, y_true: np.ndarray, preds: pd.DataFrame | np.ndarray) -> dict:
    """Standard metric row for one forecaster on one evaluation set."""
    q = np.asarray(preds, dtype=float)
    i = {tau: TAUS.index(tau) for tau in TAUS}
    return {
        "model": name,
        "pinball": round(pinball_loss(y_true, q), 6),
        "cov90": round(coverage(y_true, q[:, i[0.05]], q[:, i[0.95]]), 3),
        "cov50": round(coverage(y_true, q[:, i[0.25]], q[:, i[0.75]]), 3),
        "width90": round(float(np.mean(q[:, i[0.95]] - q[:, i[0.05]])), 5),
    }
