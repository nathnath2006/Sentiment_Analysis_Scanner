from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from sent_trader import db
from sent_trader.forecast import TAUS, baseline, dataset, metrics


def make_price_db(tmp_path, n_days=400, seed=7):
    """A synthetic ticker with a random walk, no sentiment."""
    db_path = str(tmp_path / "test.db")
    db.init_db(db_path)
    stock_id = db.add_stock("SYN", db_path)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days)))
    days = [date(2019, 1, 1) + timedelta(days=i) for i in range(n_days)]
    prices = pd.DataFrame(
        {
            "Date": [pd.Timestamp(d) for d in days],
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )
    db.add_daily_prices(stock_id, prices, db_path)
    return db_path, close, days


def test_target_is_next_day_return_and_features_are_past_only(tmp_path):
    db_path, close, days = make_price_db(tmp_path)
    frame = dataset.training_rows(dataset.build_ticker_frame("SYN", db_path=db_path))
    assert not frame.empty

    log_ret = pd.Series(np.log(close)).diff()
    day_index = {d: i for i, d in enumerate(days)}
    row = frame.iloc[10]
    t = day_index[row["date"]]
    # target = return from close(t) to close(t+1)
    assert row["target"] == pytest.approx(log_ret[t + 1])
    # most recent feature lag is day t's own return (scaled), nothing later
    assert row["ret_lag_0"] * (row["vol20"] + 1e-8) == pytest.approx(log_ret[t], rel=1e-4)


def test_baseline_quantiles_match_trailing_window(tmp_path):
    db_path, close, days = make_price_db(tmp_path)
    frame = dataset.training_rows(dataset.build_ticker_frame("SYN", db_path=db_path))
    preds = baseline.predictions(frame, 20)

    log_ret = pd.Series(np.log(close)).diff()
    day_index = {d: i for i, d in enumerate(days)}
    row_n = 5
    t = day_index[frame.iloc[row_n]["date"]]
    window = log_ret[t - 19 : t + 1]  # trailing 20 days INCLUDING day t
    assert preds.iloc[row_n][0.5] == pytest.approx(window.quantile(0.5))
    assert preds.iloc[row_n][0.05] == pytest.approx(window.quantile(0.05))


def test_pinball_loss_manual_case():
    # single sample, y=0.0; check against hand-computed pinball
    y = np.array([0.0])
    q = np.array([[-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]])
    expected = np.mean(
        [
            0.05 * 0.02,   # y above q05 by 0.02
            0.10 * 0.01,
            0.25 * 0.005,
            0.0,
            0.25 * 0.005,  # y below q75 by 0.005
            0.10 * 0.01,
            0.05 * 0.02,
        ]
    )
    assert metrics.pinball_loss(y, q) == pytest.approx(expected)


def test_coverage():
    y = np.array([0.0, 0.5, -0.5, 2.0])
    lower = np.array([-1.0, -1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])
    assert metrics.coverage(y, lower, upper) == 0.75


def test_chronological_split_no_overlap(tmp_path):
    db_path, _, _ = make_price_db(tmp_path, n_days=600)
    data = dataset.build_dataset(["SYN"], db_path=db_path)
    splits = dataset.split(data)
    assert len(splits["train"]) > 0
    if len(splits["val"]):
        assert splits["train"]["date"].max() < splits["val"]["date"].min()


def test_noncrossing_layer_orders_quantiles():
    tf = pytest.importorskip("tensorflow")
    from sent_trader.forecast import model as fmodel

    _, _, NonCrossing = fmodel._keras()
    raw = tf.constant([[0.5, -3.0, -3.0, 2.0, -1.0, 0.0, 1.0]])
    out = NonCrossing()(raw).numpy()[0]
    assert list(out) == sorted(out)
    assert out[0] == pytest.approx(0.5)
