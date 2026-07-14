"""The naive forecaster: trailing empirical quantiles of past returns.

"Tomorrow's return will be distributed like the last N days were." Zero
parameters, zero learning. Any model that cannot beat this on pinball loss
has learned nothing worth shipping — which is exactly why it exists.
"""

import pandas as pd

from . import TAUS


def predictions(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """Extract the baseline quantile forecasts computed in dataset.py."""
    cols = {f"base{window}_q{tau}": tau for tau in TAUS}
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Baseline columns missing from dataset: {missing}")
    out = data[list(cols)].copy()
    out.columns = TAUS
    return out
