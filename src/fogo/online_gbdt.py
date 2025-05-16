class OnlineGBDT:
    """
    Online Gradient Boosted Decision Trees.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    learning_rate : float
        Step size shrinkage.
    loss : callable or str
        Loss function.
    ...
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, loss='mse', **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.loss = loss

    def fit(self, X, y):
        """
        Builds model on a batch of training data.
        """
        # ...implementation...

    def fit_one(self, X, y):
        """
        Incrementally fits the model to a singular datapoint.
        """
        # ...implementation...

    def predict(self, X):
        """
        Performs prediction for a sample vector of predictors.
        """
        # ...implementation...

    def save(self, filepath):
        """Save model to disk."""
        # ...implementation...

    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        # ...implementation...