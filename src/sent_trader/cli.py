"""Command-line interface.

    sent-trader scan NVDA AAPL     collect + score one or more tickers
    sent-trader scan NVDA --model tf   ...using the trained TF classifier
    sent-trader list               show tickers in the database
    sent-trader export --outdir .  dump prices and articles to CSV
    sent-trader train              train the TF classifier on PhraseBank
    sent-trader benchmark          TF classifier vs VADER on held-out set
"""

import argparse
from pathlib import Path

from . import db, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="sent-trader", description="News sentiment + price pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_p = sub.add_parser("scan", help="Scrape news and prices for tickers, then score sentiment")
    scan_p.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. NVDA AAPL")
    scan_p.add_argument("--model", default="vader", choices=["vader", "tf"], help="Sentiment model to score with")

    sub.add_parser("list", help="List tickers stored in the database")

    export_p = sub.add_parser("export", help="Export the database to CSV files")
    export_p.add_argument("--outdir", default=".", help="Directory for the CSV files")

    backfill_p = sub.add_parser(
        "backfill",
        help="Load FNSPID historical news + full price history for the top-50 universe",
    )
    backfill_p.add_argument("--tickers", nargs="*", help="Override the default 50-ticker universe")
    backfill_p.add_argument("--model", default="tf", choices=["vader", "tf", "none"], help="Model to score the backlog with")
    backfill_p.add_argument("--max-rows", type=int, help="Read only the first N CSV rows (smoke test)")
    backfill_p.add_argument("--skip-prices", action="store_true", help="Skip the yfinance price backfill")

    score_p = sub.add_parser("score", help="Score all unscored articles with a model")
    score_p.add_argument("--model", default="tf", choices=["vader", "tf"])

    train_p = sub.add_parser("train", help="Train the TensorFlow classifier on Financial PhraseBank")
    train_p.add_argument("--epochs", type=int, default=50, help="Maximum epochs (early stopping applies)")

    sub.add_parser("benchmark", help="Evaluate the TF classifier against VADER on the held-out test set")

    ftrain_p = sub.add_parser("forecast-train", help="Train the quantile forecaster on the pooled dataset")
    ftrain_p.add_argument("--epochs", type=int, default=100)

    sub.add_parser("forecast-benchmark", help="Coverage test: forecaster vs empirical-quantile baselines")

    forecast_p = sub.add_parser("forecast", help="Predict tomorrow's return distribution for a ticker")
    forecast_p.add_argument("ticker", help="Ticker symbol, e.g. NVDA")

    sub.add_parser("app", help="Launch the Streamlit dashboard")

    args = parser.parse_args()

    if args.command == "scan":
        for ticker in args.tickers:
            result = pipeline.scan(ticker.upper(), model=args.model)
            print(
                f"{result['ticker']}: +{result['articles_added']} articles, "
                f"+{result['price_rows_added']} price rows, "
                f"{result['articles_scored']} newly scored"
            )
    elif args.command == "list":
        tickers = db.get_stock_list()
        if not tickers:
            print("Database is empty. Run: sent-trader scan <TICKER>")
        for t in tickers:
            print(t)
    elif args.command == "export":
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stock_df, articles_df = db.export_dataframes()
        stock_path = outdir / "stock_data.csv"
        articles_path = outdir / "article_data.csv"
        stock_df.to_csv(stock_path, index=False)
        articles_df.to_csv(articles_path, index=False)
        print(f"Wrote {stock_path} ({len(stock_df)} rows) and {articles_path} ({len(articles_df)} rows)")
    elif args.command == "backfill":
        result = pipeline.backfill(
            tickers=args.tickers or None,
            model=None if args.model == "none" else args.model,
            max_rows=args.max_rows,
            skip_prices=args.skip_prices,
        )
        print(
            f"Done: read {result['rows_read']:,} rows, matched {result['rows_matched']:,}, "
            f"stored {result['articles_added']:,} articles, "
            f"{result['price_rows_added']:,} price rows, scored {result['articles_scored']:,}"
        )
    elif args.command == "score":
        scored = pipeline.score_pending(model=args.model, log=print)
        print(f"Scored {scored:,} articles")
    elif args.command == "train":
        from .sentiment import tf_classifier

        result = tf_classifier.train(epochs=args.epochs)
        print(f"Saved {result['model_path']}")
        print(f"Trained on {result['train_sentences']} sentences; val accuracy {result['val_accuracy']}")
        print("Now compare against VADER: sent-trader benchmark")
    elif args.command == "benchmark":
        from .sentiment import benchmark

        benchmark.run()
    elif args.command == "forecast-train":
        from .forecast import dataset as fdataset, model as fmodel

        data = fdataset.build_dataset()
        result = fmodel.train(fdataset.split(data), epochs=args.epochs)
        print(f"Saved {result['model_path']}")
        print(f"Trained on {result['train_samples']:,} samples; val pinball {result['val_pinball']}")
        print("Now run the coverage test: sent-trader forecast-benchmark")
    elif args.command == "forecast-benchmark":
        from .forecast import evaluate

        evaluate.run()
    elif args.command == "forecast":
        import math

        from . import db as _db
        from .forecast import TAUS, dataset as fdataset, model as fmodel

        ticker = args.ticker.upper()
        row = fdataset.latest_features(ticker)
        quantiles = fmodel.predict(row).iloc[0]
        close = _db.get_stock_prices(ticker).iloc[-1]["close"]
        print(f"{ticker}: next-trading-day return distribution (from {row['date'].iloc[0]}, close {close:.2f})")
        for tau in TAUS:
            log_ret = quantiles[tau]
            pct = (math.exp(log_ret) - 1) * 100
            print(f"  q{int(tau * 100):02d}: {pct:+.2f}%  ({close * math.exp(log_ret):.2f})")
    elif args.command == "app":
        import subprocess
        import sys
        from pathlib import Path as _Path

        app_path = _Path(__file__).parent / "app.py"
        raise SystemExit(
            subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)]).returncode
        )


if __name__ == "__main__":
    main()
