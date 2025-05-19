class OnlineGBDT:

    def __init__(self, n_estimators=100, learning_rate=0.1, loss='mse', **kwargs):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.loss = loss

    def fit(self, X, y):
        """
        Builds model on a batch of training data.
        """
        print(f"Fitting model with {len(X)} samples.")
        # ...implementation...

    def fit_one(self, X, y):
        """
        Incrementally fits the model to a singular datapoint.
        """
        print(f"Fitting model with 1 sample.")
        # ...implementation...

    def predict(self, X):
        """
        Performs prediction for a sample vector of predictors.
        """
        print(f"Predicting with {len(X)} samples.")
        # ...implementation...

    def save(self, filepath):
        """Save model to disk."""
        print(f"Saving model to {filepath}.")
        # ...implementation...

    @classmethod
    def load(cls, filepath):
        """Load model from disk."""
        print(f"Loading model from {filepath}.")
        # ...implementation...