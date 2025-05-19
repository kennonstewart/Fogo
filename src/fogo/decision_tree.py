class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict(sample) for sample in X]

    def _build_tree(self, X, y, depth=0):
        # Placeholder for building the decision tree
        pass

    def _predict(self, sample):
        # Placeholder for making predictions
        pass


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.information_gain = None
        self.prediction_probabilities = None

    def is_leaf(self):  
        return self.value is not None