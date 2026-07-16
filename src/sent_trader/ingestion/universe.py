"""The default ticker universe: 50 of the largest S&P 500 companies.

Static on purpose — the training universe must not drift between runs.
Dual-class listings appear once (GOOGL, not GOOG); Berkshire is excluded
because its ticker is formatted inconsistently across data sources.
"""

TOP50 = [
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "LLY", "WMT",
    "JPM", "V", "UNH", "XOM", "ORCL", "MA", "COST", "HD", "PG", "NFLX",
    "JNJ", "ABBV", "BAC", "CRM", "KO", "AMD", "CVX", "PEP", "TMO", "CSCO",
    "WFC", "MRK", "ADBE", "LIN", "ACN", "MCD", "IBM", "GE", "ABT", "NOW",
    "PM", "TXN", "INTU", "QCOM", "CAT", "DIS", "AXP", "MS", "GS", "RTX",
]
