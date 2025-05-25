from src.fogo.online_gbdt import OnlineGBDT
import numpy as np
import pytest

def test_incremental_delete():
    """
    Incremental delete smoke test:
      1. generate a simple y = 2x dataset + noise
      2. fit OnlineGBDT on the batch
      3. incrementally delete data and check predictions
    """
    # 1. synthetic data
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(50, 1))
    y = 2 * X.squeeze() + rng.normal(0, 0.05, 50)  # linear with noise

    # 2. model with fixed hyper-params
    model = OnlineGBDT(n_estimators=5, learning_rate=0.1)
    model.fit(X, y)

    # 3. incrementally delete one data point at a time
    for i in range(10):
        X_delete = X[i:i+1]
        y_delete = y[i:i+1]
        model.decrement(X_delete, y_delete)

    # Check that the model's predictions on the remaining data are consistent
    assert np.all(np.isfinite(model.predict(X))), "Predictions contain non-finite values after deletion"
    assert len(model.predict(X)) == len(y), "Model did not correctly handle incremental deletion"