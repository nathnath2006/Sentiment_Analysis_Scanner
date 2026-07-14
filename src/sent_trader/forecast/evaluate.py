"""The coverage test: model vs baselines on the held-out years.

The claim under test, stated before any model existed: if the TF forecaster
cannot beat the trailing empirical quantiles on pinball loss, it has learned
nothing. cov90/cov50 must also sit near 0.90/0.50 — miscalibrated intervals
are worse than no intervals.
"""

import pandas as pd

from . import baseline, dataset, metrics, model


def run(db_path: str | None = None) -> list[dict]:
    print("Building dataset...")
    data = dataset.build_dataset(db_path=db_path)
    splits = dataset.split(data)
    test = splits["test"]
    y = test["target"].to_numpy()
    print(
        f"samples: train {len(splits['train']):,} / val {len(splits['val']):,}"
        f" / test {len(test):,}  (test = {test['date'].min()} .. {test['date'].max()})"
    )

    rows = [
        metrics.report(f"baseline-{w}d", y, baseline.predictions(test, w))
        for w in dataset.BASELINE_WINDOWS
    ]
    rows.append(metrics.report("tf-quantile", y, model.predict(test)))

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    best_baseline = min(r["pinball"] for r in rows if r["model"].startswith("baseline"))
    verdict = "BEATS" if rows[-1]["pinball"] < best_baseline else "DOES NOT BEAT"
    print(f"\ntf-quantile {verdict} the best baseline on pinball loss.")
    return rows
