import numpy as np

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        feat_idx, threshold, gain = self._find_best_split(X, y)

        if gain == 0:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        left_idx = X[:, feat_idx] <= threshold
        right_idx = ~left_idx

        left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return TreeNode(feature=feat_idx, threshold=threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = 0
        best_feat = None
        best_thresh = None
        current_mse = self._mse(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                mse_left = self._mse(y[left_idx])
                mse_right = self._mse(y[right_idx])
                weighted_mse = (len(y[left_idx]) * mse_left + len(y[right_idx]) * mse_right) / n_samples
                gain = current_mse - weighted_mse
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feature
                    best_thresh = threshold
        if best_gain == 0:
            return None, None, 0
        return best_feat, best_thresh, best_gain

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        X = np.asarray(X)
        preds = [self._predict_one(x, self.tree) for x in X]
        return preds

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    # ------------------------------------------------------------------ #
    # Decremental learning (machine un‑learning)
    # ------------------------------------------------------------------ #
    def decrement(self, X, residuals):
        """
        Removes the influence of the given samples from this tree.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Feature vectors of the data to forget.
        residuals : array‑like of shape (n_samples,)
            Residual targets associated with the samples to forget.
            (OnlineGBDT passes y - prediction so we re‑use that signal.)
        """
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet.")
        X = np.asarray(X)
        residuals = np.asarray(residuals)
        self._decrement_node(self.tree, X, residuals)

    def _decrement_node(self, node, X, residuals, depth=0):
        """
        Recursively walk the tree, decide whether the current split
        is still optimal after removing data, and rebuild sub‑trees if not.
        """
        if node.is_leaf():
            # leaf nodes have no split to update
            return

        cur_feat = node.feature
        cur_thresh = node.threshold

        left_idx = X[:, cur_feat] <= cur_thresh
        right_idx = ~left_idx

        # Determine the best split given the *remaining* data
        best_feat, best_thresh, best_gain = self._find_best_split(X, residuals)

        # If the optimal split has moved, rebuild this sub‑tree
        if best_gain != 0 and (
            best_feat != cur_feat or best_thresh != cur_thresh
        ):
            # Rebuild children with the new best split
            node.feature = best_feat
            node.threshold = best_thresh
            node.left = self._build_tree(
                X[left_idx], residuals[left_idx], depth + 1
            )
            node.right = self._build_tree(
                X[right_idx], residuals[right_idx], depth + 1
            )
            node.value = None  # internal nodes hold no value
        else:
            # recurse
            self._decrement_node(node.left, X[left_idx], residuals[left_idx], depth + 1)
            self._decrement_node(node.right, X[right_idx], residuals[right_idx], depth + 1)