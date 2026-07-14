"""Probabilistic next-day return forecasting.

dataset.py  builds leakage-safe samples from prices + sentiment
baseline.py trailing empirical quantiles — the bar the model must clear
model.py    TF quantile-regression net (pinball loss, non-crossing outputs)
evaluate.py pinball + coverage comparison on the held-out years
"""

# The quantiles every forecaster in this package predicts.
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
