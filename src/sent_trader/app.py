"""Streamlit front end. Launch with: sent-trader app

Four tabs: tomorrow's forecast fan, scored headlines, sentiment-vs-price
history, and the model's calibration report card.
"""

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sent_trader import db, features, pipeline
from sent_trader.forecast import TAUS, baseline, dataset, metrics
from sent_trader.forecast import model as fmodel

# Validated palette (dataviz reference instance, light surface)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
BLUE = "#2a78d6"        # categorical slot 1 / primary series
BLUE_DARK = "#1c5cab"
AQUA = "#1baf7a"        # categorical slot 2
RED = "#e34948"         # diverging negative pole
GRID = "#e8e7e3"
BAND_90 = "#cde2fb"     # sequential blue, light -> dark with confidence
BAND_50 = "#9ec5f4"

st.set_page_config(page_title="Sentiment Scanner", page_icon="📈", layout="wide")


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)


def new_figure(height=3.2):
    fig, ax = plt.subplots(figsize=(9, height))
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    return fig, ax


@st.cache_data(ttl=600)
def load_tickers():
    return db.get_stock_list()


@st.cache_data(ttl=600)
def load_prices(ticker):
    prices = db.get_stock_prices(ticker)
    prices["date"] = pd.to_datetime(prices["date"], utc=True, format="mixed").dt.date
    return prices.drop_duplicates("date").sort_values("date")


@st.cache_data(ttl=600)
def load_headlines(ticker):
    articles = db.get_article_sentiment(ticker, model_name="tf-phrasebank")
    articles["publish_date"] = pd.to_datetime(articles["publish_date"], utc=True, format="mixed")
    return articles.sort_values("publish_date", ascending=False)


@st.cache_data(ttl=600)
def load_daily_sentiment(ticker):
    return features.daily_sentiment(ticker, model_name="tf-phrasebank")


@st.cache_data(ttl=600, show_spinner="Computing forecast...")
def live_forecast(ticker):
    row = dataset.latest_features(ticker)
    quantiles = fmodel.predict(row).iloc[0]
    return row["date"].iloc[0], quantiles


@st.cache_data(ttl=3600, show_spinner="Building the evaluation dataset (about a minute)...")
def calibration_data():
    data = dataset.build_dataset()
    test = dataset.split(data)["test"]
    y = test["target"].to_numpy()
    model_preds = fmodel.predict(test)
    base_preds = baseline.predictions(test, 250)
    rows = [
        metrics.report("baseline-250d", y, base_preds),
        metrics.report("tf-quantile", y, model_preds),
    ]
    reliability = pd.DataFrame(
        {
            "tau": TAUS,
            "tf-quantile": [(y <= model_preds[t].to_numpy()).mean() for t in TAUS],
            "baseline-250d": [(y <= base_preds[t].to_numpy()).mean() for t in TAUS],
        }
    )
    return pd.DataFrame(rows), reliability, str(test["date"].min()), str(test["date"].max())


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Sentiment Scanner")
tickers = load_tickers()
if not tickers:
    st.warning("Database is empty. Run `sent-trader backfill` first.")
    st.stop()

default = tickers.index("NVDA") if "NVDA" in tickers else 0
ticker = st.sidebar.selectbox("Ticker", tickers, index=default)

if st.sidebar.button("Fetch latest news & prices"):
    with st.spinner(f"Scanning {ticker}..."):
        result = pipeline.scan(ticker, model="tf")
    st.sidebar.success(
        f"+{result['articles_added']} articles, +{result['price_rows_added']} price rows"
    )
    st.cache_data.clear()

st.sidebar.caption(
    "Sentiment: tf-phrasebank (from-scratch TF classifier). "
    "Forecast: tf-quantile v0.3 (pinball loss, vol-standardized)."
)

tab_forecast, tab_news, tab_history, tab_calibration = st.tabs(
    ["Forecast", "Headlines", "Sentiment history", "Calibration"]
)

# ---------------------------------------------------------------- forecast
with tab_forecast:
    prices = load_prices(ticker)
    close = prices.iloc[-1]["close"]
    try:
        as_of, quantiles = live_forecast(ticker)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    med = (math.exp(quantiles[0.5]) - 1) * 100
    lo, hi = ((math.exp(quantiles[t]) - 1) * 100 for t in (0.05, 0.95))
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Last close ({as_of})", f"{close:,.2f}")
    c2.metric("Median next-day move", f"{med:+.2f}%")
    c3.metric("90% interval", f"{lo:+.1f}% … {hi:+.1f}%")

    history = prices.tail(90)
    fig, ax = new_figure(3.6)
    ax.plot(history["date"], history["close"], color=BLUE, linewidth=2)
    next_x = history["date"].iloc[-1] + pd.Timedelta(days=1)
    price_q = {t: close * math.exp(quantiles[t]) for t in TAUS}
    half = pd.Timedelta(hours=14)
    for low_t, high_t, color in ((0.05, 0.95, BAND_90), (0.25, 0.75, BAND_50)):
        ax.fill_between(
            [next_x - half, next_x + half], price_q[low_t], price_q[high_t],
            color=color, linewidth=0,
        )
    ax.plot(
        [next_x - half, next_x + half], [price_q[0.5]] * 2,
        color=BLUE_DARK, linewidth=2, solid_capstyle="round",
    )
    for t, va in ((0.95, "bottom"), (0.05, "top")):
        ax.annotate(
            f" q{int(t * 100)}: {price_q[t]:,.0f}", (next_x + half, price_q[t]),
            fontsize=8, color=TEXT_2, va=va,
        )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_title(
        f"{ticker} — last 90 days, with tomorrow's forecast bands (90% / 50% / median)",
        fontsize=10, color=TEXT, loc="left",
    )
    st.pyplot(fig, width=True)

    table = pd.DataFrame(
        {
            "quantile": [f"q{int(t * 100):02d}" for t in TAUS],
            "return": [f"{(math.exp(quantiles[t]) - 1) * 100:+.2f}%" for t in TAUS],
            "price": [f"{price_q[t]:,.2f}" for t in TAUS],
        }
    )
    st.dataframe(table, hide_index=True, width=False)

# ---------------------------------------------------------------- headlines
with tab_news:
    articles = load_headlines(ticker)
    if articles.empty:
        st.info("No articles stored for this ticker yet.")
    else:
        dot = {"positive": "🔵", "negative": "🔴", "neutral": "⚪"}
        view = articles.head(40).copy()
        view["sentiment"] = view.apply(
            lambda r: f"{dot.get(r['label'], '⚪')} {r['label'] or 'unscored'} "
            f"({r['score']:+.2f})" if pd.notna(r["score"]) else "unscored",
            axis=1,
        )
        view["published"] = view["publish_date"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            view[["published", "sentiment", "title"]],
            hide_index=True, width=True, height=560,
        )

# ---------------------------------------------------------------- history
with tab_history:
    daily = load_daily_sentiment(ticker)
    prices = load_prices(ticker)
    if daily.empty:
        st.info("No scored articles for this ticker yet.")
    else:
        window = prices.tail(180)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 5), sharex=True, height_ratios=[2, 1]
        )
        fig.patch.set_facecolor(SURFACE)
        for ax in (ax1, ax2):
            style_axis(ax)
        ax1.plot(window["date"], window["close"], color=BLUE, linewidth=2)
        ax1.set_title(f"{ticker} close price", fontsize=10, color=TEXT, loc="left")

        recent = daily[daily["date"] >= window["date"].iloc[0]]
        colors = [BLUE if v >= 0 else RED for v in recent["mean_score"]]
        ax2.bar(recent["date"], recent["mean_score"], color=colors, width=1.0)
        ax2.axhline(0, color=GRID, linewidth=1)
        ax2.set_title(
            "daily mean sentiment (tf-phrasebank)", fontsize=10, color=TEXT, loc="left"
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        st.pyplot(fig, width=True)

# ------------------------------------------------------------- calibration
with tab_calibration:
    st.markdown(
        "**The honesty check.** On held-out years the model never trained on: "
        "does \"90% confident\" actually mean 90%? Pinball loss is the score "
        "being minimized; coverage should sit near its nominal level."
    )
    table, reliability, t0, t1 = calibration_data()
    st.caption(f"Held-out test period: {t0} to {t1}")
    st.dataframe(table, hide_index=True, width=False)

    fig, ax = new_figure(4.0)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax.plot(
        reliability["tau"], reliability["baseline-250d"],
        color=AQUA, linewidth=2, marker="o", markersize=6, label="baseline-250d",
    )
    ax.plot(
        reliability["tau"], reliability["tf-quantile"],
        color=BLUE, linewidth=2, marker="o", markersize=6, label="tf-quantile",
    )
    ax.set_xlabel("nominal quantile", fontsize=9, color=TEXT_2)
    ax.set_ylabel("observed frequency", fontsize=9, color=TEXT_2)
    ax.set_title(
        "Reliability: predicted quantile vs how often reality fell below it",
        fontsize=10, color=TEXT, loc="left",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    st.pyplot(fig, width=True)
