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

class OnlineGBDT:

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, random_state = None):
        """
        Initialize the Online Gradient Boosting Decision Tree (GBDT) model.
        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the ensemble.
        learning_rate : float, default=0.1
            The learning rate shrinks the contribution of each tree.
        max_depth : int, default=3
            The maximum depth of the individual trees.
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        random_state : int, optional
            Controls the randomness of the estimator. Pass an int for reproducible results across multiple function calls.
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = None
        self.trees = []


    def set_params(self, **params):
        """
        Set the parameters of this estimator. Compatible with scikit-learn's API.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key} for OnlineGBDT.")
        return self


    def fit(self, X, y):
        """
        Fit an ensemble of decision‑trees on the initial batch.
        Standard gradient‑boosting: each tree trains on the residuals
        left by its predecessors.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
        y : array‑like of shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        n_samples = X.shape[0]
        # Start with zero prediction
        pred = np.zeros(n_samples, dtype=float)
        self.trees = []

        for _ in range(self.n_estimators):
            residuals = y - pred
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)

            update = np.array(tree.predict(X))
            pred += self.learning_rate * update
            self.trees.append(tree)

    def predict(self, X):
        """
        Sum the (learning‑rate‑scaled) predictions of all trees.
        """
        X = np.asarray(X)
        if not self.trees:
            raise ValueError("Model has no trees. Call `fit` first.")
        agg = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            agg += self.learning_rate * np.array(tree.predict(X))
        return agg


    def fit_one(self, x, y):
        """
        Incrementally update the existing ensemble **in‑place** without
        adding new trees.  We traverse every tree, find the leaf reached
        by the sample, and nudge its prediction toward the new target.

        Parameters
        ----------
        x : array‑like of shape (n_features,)  or (1, n_features)
        y : float or int
        """
        x = np.asarray(x).reshape(1, -1)
        y = float(y)

        # Current ensemble prediction
        current_pred = self.predict(x)[0]
        residual = y - current_pred

        # Distribute the residual across trees (simple shrinkage)
        step = (self.learning_rate * residual) / len(self.trees)

        for tree in self.trees:
            leaf = self._find_leaf(tree.tree, x[0])
            # Update leaf value in‑place
            leaf.value += step
    
    def decrement(self, x, residual):
        """
        Remove the influence of a single sample from the ensemble.

        Parameters
        ----------
        x : array‑like of shape (n_features,)  or (1, n_features)
        residual : float or int
            The residual target associated with the sample to forget.
            (OnlineGBDT passes y - prediction so we re‑use that signal.)
        """
        x = np.asarray(x).reshape(1, -1)
        residual = float(residual)

        for tree in self.trees:
            leaf = self._find_leaf(tree.tree, x[0])
            # Update leaf value in‑place
            leaf.value -= residual
            # Rebuild the tree if necessary
            if leaf.value == 0:
                # If the leaf value is zero, we need to rebuild the tree
                # This is a simplified version; in practice, you might want
                # to handle this differently.
                self._decrement_node(tree.tree, x[0], residual)
                # Rebuild the tree
                tree.fit(x, np.array([residual]))
                # Reset the leaf value
                leaf.value = np.mean(residual)
                # Note: This is a simplified approach. In practice, you might
                # want to handle this differently, especially if the tree
                # structure changes significantly.

    def score(self, X, y):
        """
        Compute the coefficient of determination R^2 manually.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def get_params(self, deep=True):
        """
        Return estimator parameters as a dict. Required for scikit-learn compatibility.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
        }

    def _decrement_node(self, node, x_row, residual):
        """
        Recursively walk the tree, decide whether the current split
        is still optimal after removing data, and rebuild sub‑trees if not.
        """
        if node.is_leaf():
            # leaf nodes have no split to update
            return

        cur_feat = node.feature
        cur_thresh = node.threshold

        if x_row[cur_feat] <= cur_thresh:
            self._decrement_node(node.left, x_row, residual)
        else:
            self._decrement_node(node.right, x_row, residual)
        # Determine the best split given the *remaining* data
        best_feat, best_thresh, best_gain = self._find_best_split(x_row, residual)
        # If the optimal split has moved, rebuild this sub‑tree
        if best_gain != 0 and (
            best_feat != cur_feat or best_thresh != cur_thresh
        ):
            # Rebuild children with the new best split
            node.feature = best_feat
            node.threshold = best_thresh
            node.left = self._build_tree(
                x_row, residual, depth=1
            )
            node.right = self._build_tree(
                x_row, residual, depth=1
            )
            node.value = None
        else:
            # recurse
            self._decrement_node(node.left, x_row, residual)
            self._decrement_node(node.right, x_row, residual)
        # Note: This is a simplified approach. In practice, you might
        # want to handle this differently, especially if the tree
        # structure changes significantly.

    def _find_leaf(self, node, x_row):
        """
        Follow the decision path for x_row and return the leaf node object.
        """
        if node.is_leaf():
            return node
        if x_row[node.feature] <= node.threshold:
            return self._find_leaf(node.left, x_row)
        else:
            return self._find_leaf(node.right, x_row)