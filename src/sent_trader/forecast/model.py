"""TensorFlow quantile-regression forecaster.

Small MLP trained with pinball loss on all quantiles at once. The final
layer predicts the lowest quantile plus softplus increments, so predicted
quantiles can never cross. Trained on ~100k pooled samples; depth would
only buy memorization here.

The model predicts quantiles of the VOL-STANDARDIZED return (y / vol20);
predictions are rescaled by each day's trailing vol. One pooled network
can't otherwise emit correct scales for both a calm mega-cap and a 2022
drawdown — the vol term carries the scale, the net learns the shape.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import config
from . import TAUS
from .dataset import FEATURE_COLS

MODEL_NAME = "tf-quantile"
MODEL_VERSION = "0.3.0"

VOL_EPS = 1e-8

_model = None


def model_path() -> Path:
    return Path(config.models_dir) / f"{MODEL_NAME}-{MODEL_VERSION}.keras"


def _keras():
    import keras

    @keras.saving.register_keras_serializable(package="sent_trader")
    def pinball(y_true, y_pred):
        import tensorflow as tf

        taus = tf.constant(TAUS, dtype=y_pred.dtype)
        diff = tf.reshape(y_true, [-1, 1]) - y_pred  # broadcast over quantiles
        return tf.reduce_mean(tf.maximum(taus * diff, (taus - 1.0) * diff))

    @keras.saving.register_keras_serializable(package="sent_trader")
    class NonCrossing(keras.layers.Layer):
        """Map raw outputs to ordered quantiles: q0, then +softplus steps."""

        def call(self, x):
            import tensorflow as tf

            base = x[:, :1]
            steps = tf.nn.softplus(x[:, 1:])
            return tf.concat([base, base + tf.cumsum(steps, axis=1)], axis=1)

    return keras, pinball, NonCrossing


def build_model(train_features: np.ndarray):
    keras, pinball, NonCrossing = _keras()
    from keras import layers

    norm = layers.Normalization()
    norm.adapt(train_features)

    model = keras.Sequential(
        [
            keras.Input(shape=(len(FEATURE_COLS),)),
            norm,
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(len(TAUS)),
            NonCrossing(),
        ]
    )
    model.compile(optimizer="adam", loss=pinball)
    return model


def train(splits: dict[str, pd.DataFrame], epochs: int = 100, verbose: int = 2) -> dict:
    keras, _, _ = _keras()
    keras.utils.set_random_seed(config.seed)

    def xy(part: pd.DataFrame):
        x = part[FEATURE_COLS].to_numpy(dtype=np.float32)
        vol = part["vol20"].to_numpy(dtype=np.float32) + VOL_EPS
        y = part["target"].to_numpy(dtype=np.float32) / vol
        return x, y

    x_train, y_train = xy(splits["train"])
    x_val, y_val = xy(splits["val"])

    model = build_model(x_train)
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=256,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
        ],
        verbose=verbose,
    )

    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

    global _model
    _model = model
    return {
        "model_path": str(path),
        "train_samples": len(x_train),
        "val_pinball": round(float(model.evaluate(x_val, y_val, verbose=0)), 6),
    }


def _get_model():
    global _model
    if _model is None:
        keras, _, _ = _keras()  # registers custom objects before loading
        path = model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"No trained forecaster at {path}. Run: sent-trader forecast-train"
            )
        _model = keras.models.load_model(path)
    return _model


def predict(data: pd.DataFrame) -> pd.DataFrame:
    """Quantile forecasts for each dataset row; columns are TAUS.

    Rescales the net's standardized quantiles by each row's trailing vol.
    """
    x = data[FEATURE_COLS].to_numpy(dtype=np.float32)
    vol = data["vol20"].to_numpy(dtype=np.float32) + VOL_EPS
    preds = _get_model().predict(x, verbose=0) * vol[:, None]
    return pd.DataFrame(preds, columns=TAUS, index=data.index)
