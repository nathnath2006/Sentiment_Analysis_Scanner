"""Daily price ingestion via yfinance."""

import pandas as pd
import yfinance as yf

from ..config import config


def fetch_daily_prices(ticker: str, period: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV bars for a ticker.

    Returns a DataFrame with columns Date, Open, High, Low, Close, Volume,
    one row per trading day, ready for db.add_daily_prices.
    """
    raw = yf.download(
        ticker,
        period=period or config.price_period,
        interval=config.price_interval,
        multi_level_index=False,
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    return raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
