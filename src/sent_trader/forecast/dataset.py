"""Build the supervised dataset: one sample per (ticker, trading day).

Leakage rules, enforced here and nowhere else:
- The target for day t is the log return from close(t) to close(t+1).
- Every feature for day t uses information available at close(t): returns
  up to t, volatility up to t, and sentiment whose *effective date* is <= t
  (features.effective_date already pushed after-close headlines forward).
- Splits are chronological. Never shuffle before splitting.

Samples are restricted to the FNSPID news era so train/val/test share the
same feature regime.
"""

import numpy as np
import pandas as pd

from .. import db, features
from ..ingestion.universe import TOP50
from . import TAUS

LOOKBACK = 20          # trading days of return history per sample
BASELINE_WINDOWS = (20, 250)
MIN_HISTORY = 250      # so baselines and model see the same sample set
DATA_START = "2009-01-01"
DATA_END = "2023-12-31"
TRAIN_END = "2019-12-31"
VAL_END = "2021-12-31"

RETURN_COLS = [f"ret_lag_{i}" for i in range(LOOKBACK)]
SENTIMENT_COLS = ["sent_mean", "sent_count", "sent_std", "sent_mean_5d"]
FEATURE_COLS = RETURN_COLS + ["vol20"] + SENTIMENT_COLS


def build_ticker_frame(
    ticker: str, model_name: str = "tf-phrasebank", db_path: str | None = None
) -> pd.DataFrame:
    """Per-ticker daily frame with target, features and baseline quantiles."""
    prices = db.get_stock_prices(ticker, db_path)
    if len(prices) < MIN_HISTORY + 2:
        return pd.DataFrame()
    prices["date"] = pd.to_datetime(prices["date"], utc=True, format="mixed").dt.date
    prices = prices.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    frame = pd.DataFrame({"date": prices["date"]})
    frame["log_ret"] = prices['close'].apply(np.log).diff()
    frame["vol20"] = frame["log_ret"].rolling(LOOKBACK).std()

    # return lags, scaled by trailing vol so tickers pool on one scale;
    # ret_lag_0 is the most recent day
    scale = frame["vol20"] + 1e-8
    for i in range(LOOKBACK):
        frame[f"ret_lag_{i}"] = frame["log_ret"].shift(i) / scale

    # sentiment effective on day t (known by close of t)
    sent = features.daily_sentiment(ticker, model_name, db_path)
    if sent.empty:
        frame[["sent_mean", "sent_count", "sent_std"]] = 0.0
    else:
        sent = sent.rename(
            columns={"mean_score": "sent_mean", "article_count": "sent_count", "score_std": "sent_std"}
        )
        frame = frame.merge(sent, on="date", how="left")
        frame[["sent_mean", "sent_count", "sent_std"]] = frame[
            ["sent_mean", "sent_count", "sent_std"]
        ].fillna(0.0)
    frame["sent_count"] = np.log1p(frame["sent_count"])
    frame["sent_mean_5d"] = frame["sent_mean"].rolling(5, min_periods=1).mean()

    # trailing empirical quantiles = the baseline forecasts for day t+1
    for window in BASELINE_WINDOWS:
        rolling = frame["log_ret"].rolling(window)
        for tau in TAUS:
            frame[f"base{window}_q{tau}"] = rolling.quantile(tau)

    frame["target"] = frame["log_ret"].shift(-1)
    frame["ticker"] = ticker

    frame = frame.iloc[MIN_HISTORY:]  # full history for every baseline window
    frame = frame[(frame["date"] >= pd.Timestamp(DATA_START).date())]
    return frame.reset_index(drop=True)


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for supervised training (features and target complete)."""
    return frame.dropna(subset=FEATURE_COLS + ["target"]).reset_index(drop=True)


def latest_features(
    ticker: str, model_name: str = "tf-phrasebank", db_path: str | None = None
) -> pd.DataFrame:
    """The most recent feature-complete row for a ticker — the live forecast
    input. Its target is NaN by definition (tomorrow hasn't happened)."""
    frame = build_ticker_frame(ticker, model_name, db_path)
    frame = frame.dropna(subset=FEATURE_COLS)
    if frame.empty:
        raise ValueError(f"Not enough stored history to build features for {ticker}")
    return frame.iloc[-1:]


def build_dataset(
    tickers: list[str] | None = None,
    model_name: str = "tf-phrasebank",
    db_path: str | None = None,
    end: str | None = DATA_END,
) -> pd.DataFrame:
    frames = [
        f
        for t in (tickers or TOP50)
        if not (f := training_rows(build_ticker_frame(t, model_name, db_path))).empty
    ]
    data = pd.concat(frames, ignore_index=True)
    if end:
        data = data[data["date"] <= pd.Timestamp(end).date()]
    return data.sort_values(["date", "ticker"]).reset_index(drop=True)


def split(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Chronological train/val/test split."""
    train_end = pd.Timestamp(TRAIN_END).date()
    val_end = pd.Timestamp(VAL_END).date()
    return {
        "train": data[data["date"] <= train_end],
        "val": data[(data["date"] > train_end) & (data["date"] <= val_end)],
        "test": data[data["date"] > val_end],
    }
