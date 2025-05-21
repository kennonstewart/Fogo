import numpy as np
import pytest

from src.fogo.online_gbdt import OnlineGBDT

@pytest.fixture
def small_dataset():
    """Initial training batch and helper points."""
    X_init = np.array([[1.0], [2.0], [3.0]])
    y_init = np.array([1.0, 2.0, 3.0])

    X_inc  = np.array([4.0])      # single-point increment
    y_inc  = np.array([4.0])

    X_del  = np.array([[2.0]])    # point to delete (same shape as predict expects)
    y_del  = np.array([2.0])

    return X_init, y_init, X_inc, y_inc, X_del, y_del


def test_increment_and_decrement(small_dataset):
    """
    1.  Train on a small batch.
    2.  Incrementally add one sample (fit_one).
    3.  Decrementally remove one sample (delete).
    4.  Verify predictions & internal state change as expected.
    """

    X_init, y_init, X_inc, y_inc, X_del, y_del = small_dataset

    # ---- initial batch fit -------------------------------------------------
    model = OnlineGBDT(n_estimators=3, learning_rate=0.1)
    model.fit(X_init, y_init)
    preds_after_fit = model.predict(X_init)

    # ---- incremental update ------------------------------------------------
    n_trees_before_inc = len(model.trees)
    model.fit_one(X_inc, y_inc)
    n_trees_after_inc  = len(model.trees)
    preds_after_inc    = model.predict(np.vstack([X_init, X_inc.reshape(1, -1)]))

    # sanity: a new tree should be added
    assert n_trees_after_inc == n_trees_before_inc + 1

    # predictions should have changed for at least one sample
    assert not np.allclose(preds_after_fit, preds_after_inc[: len(y_init)])

    # ---- decremental unlearning -------------------------------------------
    model.delete(X_del, y_del)
    preds_after_del = model.predict(np.vstack([X_init, X_inc.reshape(1, -1)]))

    # predictions should differ after deletion
    assert not np.allclose(preds_after_inc, preds_after_del)

    # model should still return float predictions of correct length
    assert preds_after_del.shape[0] == len(X_init) + 1
    assert preds_after_del.dtype.kind == "f"