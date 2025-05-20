import pickle
from fogo.decision_tree import DecisionTree
from fogo.decision_tree import TreeNode
import numpy as np


class OnlineGBDT:

    def __init__(self, n_estimators=100, learning_rate=0.1, loss='mse', **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.loss = loss

    def fit(self, X, y):
        """
        Builds model on a batch of training data using gradient boosting.
        """
        print(f"Fitting model with {len(X)} sample(s).")
        X = np.array(X)
        y = np.array(y)
        predictions = np.zeros_like(y, dtype=float)
        self.trees = []
        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = DecisionTree(max_depth=3)
            tree.fit(X, residuals)
            update = np.array(tree.predict(X))
            predictions += self.learning_rate * update
            self.trees.append(tree)

    def fit_one(self, X, y):
        """
        Incrementally fits the model to a single datapoint.
        """
        from .decision_tree import DecisionTree # why am I importing this internally?
        print(f"Fitting model with 1 sample(s).")
        # Ensure X is a single sample (1D feature vector)
        if not self.trees:
            initial_tree = DecisionTree(max_depth=3)
            initial_tree.fit([X], [y])
            self.trees.append(initial_tree)
            return 

        pred = sum(self.learning_rate * tree.predict([X])[0] for tree in self.trees)
        residual = y - pred
        new_tree = DecisionTree(max_depth=3)
        new_tree.fit([X], [residual])
        self.trees.append(new_tree)

    def predict(self, X):
        """
        Performs prediction for a sample vector of predictors.
        """
        print(f"Predicting with {len(X)} sample(s).")
        X = np.array(X)
        predictions = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            predictions += self.learning_rate * np.array(tree.predict(X))
        return predictions

    def save(self, filepath):
        """Save model to disk."""
        print(f"Saving model to {filepath}.")
        with open(filepath, 'wb') as f:
            import pickle
            pickle.dump(self, f)
        print("Model saved successfully.")

    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        print(f"Loading model from {filepath}.")
        with open(filepath, 'rb') as f:
            import pickle
            model = pickle.load(f)
        print("Model loaded successfully.")
        return model
    def delete(self, X, y):
        """
        Decrementally unlearns a datapoint or batch from the model.
        """
        print(f"Deleting {len(X)} sample(s) from model.")
        X = np.array(X)
        y = np.array(y)
        for tree in self.trees:
            residuals = y - np.array(tree.predict(X))
            tree.decrement(X, residuals)