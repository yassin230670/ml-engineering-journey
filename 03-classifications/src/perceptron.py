import numpy as np


class Perceptron:
    """Classic online binary Perceptron for labels encoded as 0 and 1."""

    def __init__(self, learning_rate=0.1, max_epochs=100, random_state=42,
                 shuffle=True, early_stopping=True):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.random_state = random_state
        self.shuffle = shuffle
        self.early_stopping = early_stopping

    def _validate_X(self, X, fitting=False):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty 2D array.")
        if not np.isfinite(X).all():
            raise ValueError("X contains missing or non-finite values.")
        if not fitting and hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return X

    def fit(self, X, y):
        #prepare x and y 
        X = self._validate_X(X, fitting=True)
        y = np.asarray(y)
        #error handeling
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array with one label per row of X.")
        if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
            raise ValueError("y must contain both binary labels 0 and 1.")
        y = y.astype(int)
        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(loc=0.0, scale=0.01, size=self.n_features_in_)
        self.bias_ = 0.0
        self.mistakes_history_ = []
        self.accuracy_history_ = []
        self.weights_history_ = [self.weights_.copy()]
        self.bias_history_ = [self.bias_]
        self.converged_ = False

        for _ in range(self.max_epochs):
            indices = np.arange(len(X))
            if self.shuffle:
                rng.shuffle(indices)
            mistakes = 0
            #iterate over samples and update by each one 
            for i in indices:
                score = float(np.dot(X[i], self.weights_) + self.bias_)
                prediction = 1 if score >= 0 else 0
                error = y[i] - prediction
                if error != 0:
                    self.weights_ += self.learning_rate * error * X[i]
                    self.bias_ += self.learning_rate * error
                    mistakes += 1
            #make a prediction, store (mistakes, accuracy, whights, bias)
            predictions = self.predict(X)
            self.mistakes_history_.append(mistakes)
            self.accuracy_history_.append(float(np.mean(predictions == y)))
            self.weights_history_.append(self.weights_.copy())
            self.bias_history_.append(float(self.bias_))
            #checking convergence 
            if mistakes == 0:
                self.converged_ = True
                if self.early_stopping:
                    break
        self.n_epochs_ = len(self.mistakes_history_)
        return self

    def net_input(self, X):
        X = self._validate_X(X)
        return X @ self.weights_ + self.bias_

    def decision_function(self, X):
        return self.net_input(X)

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def evaluate(self, X, y):
        y = np.asarray(y, dtype=int)
        predictions = self.predict(X)
        if y.shape != predictions.shape:
            raise ValueError("y must have one label per row of X.")
        return {"accuracy": float(np.mean(predictions == y)),
                "predictions": predictions}
