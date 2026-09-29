import numpy as np
import pandas as pd


def train_linear_regression(X, y, lr=0.1, epochs=2000):
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0.0

    for epoch in range(epochs):
        y_pred = X @ weights + bias
        error = y_pred - y

        dw = (X.T @ error) / n
        db = np.mean(error)

        weights -= lr * dw
        bias -= lr * db

        if epoch % 200 == 0 or epoch == epochs - 1:
            cost = np.mean(error ** 2) / 2
            print(f"Epoch {epoch:5d} | Cost: {cost:,.2f}")

    return weights, bias


def predict(X, weights, bias):
    return X @ weights + bias


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


if __name__ == "__main__":
    df = pd.read_csv("car_data.csv")

    X = df[["year", "km_driven"]].to_numpy(dtype=float)
    y = df["selling_price"].to_numpy(dtype=float)

    # Standardize features (needed for gradient descent to converge well)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    # Train/test split
    np.random.seed(42)
    idx = np.random.permutation(len(y))
    split = int(len(y) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Training...\n")
    weights, bias = train_linear_regression(X_train, y_train, lr=0.1, epochs=2000)

    print(f"\nLearned weights (year, km_driven): {weights}")
    print(f"Bias: {bias:,.2f}")

    train_pred = predict(X_train, weights, bias)
    test_pred = predict(X_test, weights, bias)

    print(f"\nTrain R^2: {r_squared(y_train, train_pred):.4f} | RMSE: {rmse(y_train, train_pred):,.2f}")
    print(f"Test  R^2: {r_squared(y_test, test_pred):.4f} | RMSE: {rmse(y_test, test_pred):,.2f}")

    # Example prediction
    example = np.array([[2015, 40000]], dtype=float)
    example_scaled = (example - X_mean) / X_std
    pred_price = predict(example_scaled, weights, bias)[0]
    print(f"\nExample: year=2015, km_driven=40000 -> predicted price ≈ {pred_price:,.2f}")