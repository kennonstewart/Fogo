import numpy as np
from src.fogo.decision_tree import DecisionTree

def test_fit_and_predict():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 1, 2, 3])
    
    model = DecisionTree(max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert isinstance(predictions, list) or isinstance(predictions, np.ndarray)
    assert all(predictions[i] == y[i] for i in range(len(y)))
    assert model.tree is not None
    assert model.tree.value is None  # Root is an internal node, not a leaf