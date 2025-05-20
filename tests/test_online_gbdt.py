import numpy as np
from src.fogo.online_gbdt import OnlineGBDT

def test_fit_and_predict_batch():
    X = np.random.rand(100, 3)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, size=100)
    
    model = OnlineGBDT(n_estimators=10)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert isinstance(predictions, list) or isinstance(predictions, np.ndarray)

def test_fit_one_predict():
    X = np.array([[0.1, 0.2, 0.3]])
    y = np.array([0.6])
    
    model = OnlineGBDT(n_estimators=1)
    model.fit_one(X[0], y[0])
    pred = model.predict(X)

    assert len(pred) == 1