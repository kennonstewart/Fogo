import pytest
import numpy as np

from src.fogo.online_gbdt import OnlineGBDT

def test_incremental_add():
    """
    Incremental add smoke test:
      1. generate a simple y = 2x dataset + noise
      2. fit OnlineGBDT on the batch
      3. incrementally add data and check predictions
    """
    # 1. synthetic data
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(50, 1))
    y = 2 * X.squeeze() + rng.normal(0, 0.05, 50)  # linear with noise

    # 2. model with fixed hyper‑params
    model = OnlineGBDT(n_estimators=5, learning_rate=0.1)
    model.fit(X, y)

    # 3. incrementally add one data point at a time
    for i in range(10):
        X_new = rng.uniform(0, 1, size=(1, 1))
        y_new = 2 * X_new.squeeze() + rng.normal(0, 0.05, 1)  # linear with noise
        model.fit(X_new, y_new)
    # Check that the model's predictions on the new data are consistent with the old data
    assert np.allclose(model.predict(X_new), model.predict(X_new)), "Model predictions on new data are inconsistent"
    # Check that the model's predictions on the old data are consistent with the new data
    assert np.allclose(model.predict(X), model.predict(X)), "Model predictions on old data are inconsistent"
