


import numpy as np
import pytest

from src.fogo.online_gbdt import OnlineGBDT

def test_batch_fit_small_dataset():
    """
    Batch‑fit smoke test:
      1. generate a simple y = 2x dataset + noise
      2. fit OnlineGBDT on the batch
      3. ensure predictions are finite and MSE is lower than naive baseline
    """
    # 1. synthetic data
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(50, 1))
    y = 2 * X.squeeze() + rng.normal(0, 0.05, 50)  # linear with noise

    # 2. model with fixed hyper‑params
    model = OnlineGBDT(n_estimators=5, learning_rate=0.1)
    model.fit(X, y)

    # 3. predictions + basic quality
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds)), "Predictions contain non‑finite values"

    mse_model  = np.mean((preds - y) ** 2)
    mse_naive  = np.mean((np.mean(y) - y) ** 2)

    # model should beat naive mean baseline
    assert mse_model < mse_naive, "Model MSE did not improve over naive baseline"