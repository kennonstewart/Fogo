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
    assert model.tree.value == 1.5  # Check the value of the root node
    assert model.tree.feature == 0  # Check the feature used for the split
    assert model.tree.threshold == 1.5  # Check the threshold used for the split
    assert model.tree.left.value == 0.5  # Check the value of the left child
    assert model.tree.right.value == 2.5  # Check the value of the right child
    assert model.tree.left.feature is None  # Check that the left child is a leaf
    assert model.tree.right.feature is None  # Check that the right child is a leaf
    assert model.tree.left.left is None  # Check that the left child has no children
    assert model.tree.left.right is None  # Check that the left child has no children
    assert model.tree.right.left is None  # Check that the right child has no children
    assert model.tree.right.right is None  # Check that the right child has no children 