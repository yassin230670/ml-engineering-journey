import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def predict(self, X):
        return X @ self.weights + self.bias

    def mean_squared_error(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y):
        #Train the model using Batch Gradient Descent.

        # Number of training examples and features
        m, n = X.shape

        # Initialize parameters
        self.weights = np.zeros((n, 1))
        self.bias = 0.0

        # Convert y to column vector
        y = y.reshape(-1, 1)

        for epoch in range(self.epochs):

            # 1. Forward Pass
            y_pred = self.predict(X)

            # 2. Calculate Error
            errors = y_pred - y

            # 3. Calculate Gradients
            dw = (2 / m) * (X.T @ errors)
            db = (2 / m) * np.sum(errors)

            # 4. Update Parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 5. Calculate MSE
            mse = self.mean_squared_error(y, y_pred)
            self.loss_history.append(mse)

    def r2_score(self, y, y_pred):
        y = y.reshape(-1, 1)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)



# Example: Student Study Hours → Exam Score

if __name__ == "__main__":

    # Dataset
    study_hours = np.array([
        2,
        4,
        6,
        8,
        10
    ], dtype=float).reshape(-1, 1)

    scores = np.array([
        50,
        65,
        78,
        90,
        98
    ], dtype=float)

    # Create model
    model = LinearRegression(
        learning_rate=0.001,
        epochs=5000
    )

    # Train model
    model.fit(study_hours, scores)

    # Predictions
    predictions = model.predict(study_hours)

    # Evaluation
    mse = model.mean_squared_error(
        scores.reshape(-1, 1),
        predictions
    )

    r2 = model.r2_score(
        scores,
        predictions
    )

    # Results
    print("Linear Regression Results")
    print("-" * 30)

    print(f"Weight (w): {model.weights.item():.4f}")
    print(f"Bias (b):   {model.bias:.4f}")
    print(f"MSE:        {mse:.4f}")
    print(f"R²:         {r2:.4f}")

    # Predict score for a new student
    new_student = np.array([[7.0]])

    predicted_score = model.predict(new_student)

    print()
    print(
        f"Predicted score for 7 study hours: "
        f"{predicted_score.item():.2f}"
    )