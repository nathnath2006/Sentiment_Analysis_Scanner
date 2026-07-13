"""Central configuration.

Every tunable lives here, resolved from environment variables with sane
defaults, so no module ever hardcodes a path or URL.
"""

import os
from dataclasses import dataclass, field


def _default_db_path() -> str:
    return os.getenv("SENT_TRADER_DB", "sentiment.db")


@dataclass
class Config:
    db_path: str = field(default_factory=_default_db_path)
    news_feed_url: str = "https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    price_period: str = "1mo"
    price_interval: str = "1d"
    # ML: dataset cache, trained-model artifacts, and the seed that makes
    # train/val/test splits reproducible across runs.
    data_dir: str = field(default_factory=lambda: os.getenv("SENT_TRADER_DATA_DIR", "data"))
    models_dir: str = field(default_factory=lambda: os.getenv("SENT_TRADER_MODELS_DIR", "models"))
    seed: int = 42


# Single shared instance; import this rather than building your own.
config = Config()
