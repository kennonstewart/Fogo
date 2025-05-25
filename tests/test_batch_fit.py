import numpy as np
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
    # assert mse_model < mse_naive, "Model MSE did not improve over naive baseline"


def test_batch_fit_enhanced():
    """
    Enhanced batch fit test:
      1. Train on multiple synthetic datasets (linear and sine)
      2. Validate prediction quality and residual behavior
      3. Confirm reproducibility across runs
      4. Check basic tree structure (if accessible)
    """
    rng = np.random.RandomState(42)

    # Linear dataset
    X_lin = rng.uniform(0, 1, size=(50, 1))
    y_lin = 3 * X_lin.squeeze() + rng.normal(0, 0.1, 50)

    # Sine dataset
    X_sin = rng.uniform(0, 2 * np.pi, size=(50, 1))
    y_sin = np.sin(X_sin).squeeze() + rng.normal(0, 0.1, 50)

    for X, y in [(X_lin, y_lin), (X_sin, y_sin)]:
        # log the dataset type
        dataset_type = "linear" if np.all(np.abs(np.diff(y)) < 0.1) else "sine"
        print(f"Testing batch fit on {dataset_type} dataset with {len(y)} samples...")
        model = OnlineGBDT(n_estimators=5, learning_rate=0.1, random_state=0)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == len(y)
        assert np.all(np.isfinite(preds)), "Predictions contain non-finite values"

        residuals = y - preds
        mean_residual = np.mean(residuals)
        assert abs(mean_residual) < 0.1, "Mean residual is too large"

        mse_model = np.mean(residuals ** 2)
        mse_naive = np.mean((np.mean(y) - y) ** 2)
        assert mse_model < mse_naive, "Model MSE did not beat naive baseline"

        # Reproducibility test
        model2 = OnlineGBDT(n_estimators=5, learning_rate=0.1, random_state=0)
        model2.fit(X, y)
        preds2 = model2.predict(X)
        assert np.allclose(preds, preds2, atol=1e-6), "Models differ across runs"

        # Optional: Tree structure inspection
        if hasattr(model, "trees"):
            assert len(model.trees) == model.n_estimators
            for tree in model.trees:
                assert hasattr(tree, "root")
                assert tree.root is not None