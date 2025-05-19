import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, loss=None):
        self.max_depth = max_depth # maximum depth of the tree
        self.min_samples_split = min_samples_split # minimum number of samples required to split an internal node
        self.loss = loss if loss is not None else self._mse # loss function, with MSE as the default
        self.tree = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        X = np.array(X)
        return [self._predict(sample, self.tree) for sample in X]

    def _build_tree(self, X, y, depth=0):
        # stopping criteria: if the tree exceeds parameters
        depth_exceeds_max = self.max_depth is not None and depth >= self.max_depth
        minimum_samples_reached = len(y) < self.min_samples_split

        if depth_exceeds_max or minimum_samples_reached or np.var(y) == 0:
            leaf_value = np.mean(y)
            return TreeNode(value=leaf_value)

        # find the best split
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        # stopping criteria: if there's no information gain 
        if best_gain == 0:
            return TreeNode(value=np.mean(y))

        # perform the split 
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gain = self._information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def _information_gain(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        gain = np.var(parent) - (weight_left * np.var(left) + weight_right * np.var(right))
        return gain

    def _predict(self, sample, node):
        if node.is_leaf():
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict(sample, node.left)
        else:
            return self._predict(sample, node.right)

    def _mse(self, y):
        return np.var(y)


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