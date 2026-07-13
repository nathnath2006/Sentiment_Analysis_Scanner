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

    train_p = sub.add_parser("train", help="Train the TensorFlow classifier on Financial PhraseBank")
    train_p.add_argument("--epochs", type=int, default=50, help="Maximum epochs (early stopping applies)")

    sub.add_parser("benchmark", help="Evaluate the TF classifier against VADER on the held-out test set")

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
    elif args.command == "train":
        from .sentiment import tf_classifier

        result = tf_classifier.train(epochs=args.epochs)
        print(f"Saved {result['model_path']}")
        print(f"Trained on {result['train_sentences']} sentences; val accuracy {result['val_accuracy']}")
        print("Now compare against VADER: sent-trader benchmark")
    elif args.command == "benchmark":
        from .sentiment import benchmark

        benchmark.run()


if __name__ == "__main__":
    main()
