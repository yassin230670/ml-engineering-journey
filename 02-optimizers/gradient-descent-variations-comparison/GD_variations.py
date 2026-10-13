
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "california_housing.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]
TARGET = "MedHouseVal"

df = pd.read_csv(DATA_PATH)
df = df.dropna().reset_index(drop=True)

X = df[FEATURES].to_numpy(dtype=np.float64)
y = df[TARGET].to_numpy(dtype=np.float64).reshape(-1, 1)

n = X.shape[0]
idx = np.random.permutation(n)
split = int(0.8 * n)
train_idx, test_idx = idx[:split], idx[split:]

X_train_raw, X_test_raw = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

p_low = np.percentile(X_train_raw, 1, axis=0)
p_high = np.percentile(X_train_raw, 99, axis=0)
X_train_raw = np.clip(X_train_raw, p_low, p_high)
X_test_raw = np.clip(X_test_raw, p_low, p_high)

mu = X_train_raw.mean(axis=0)
sigma = X_train_raw.std(axis=0)
sigma[sigma == 0] = 1.0

X_train = (X_train_raw - mu) / sigma
X_test = (X_test_raw - mu) / sigma

X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

n_features = X_train_b.shape[1]

print(f"Dataset rows: {n} | Train: {X_train_b.shape[0]} | Test: {X_test_b.shape[0]}")
print(f"Features used ({len(FEATURES)}): {FEATURES}")


def predict(X, theta):
    return X @ theta


def compute_mse_cost(X, y, theta):
    m = X.shape[0]
    errors = predict(X, theta) - y
    return float(((errors.T @ errors) / m).item())


def compute_gradient(X, y, theta):
    m = X.shape[0]
    errors = predict(X, theta) - y
    return (2.0 / m) * (X.T @ errors)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot


def batch_gradient_descent(X, y, lr=0.1, epochs=500, tol=1e-7):
    m, n_feat = X.shape
    theta = np.zeros((n_feat, 1))
    cost_history = []
    prev_cost = np.inf

    start = time.perf_counter()

    for epoch in range(epochs):
        grad = compute_gradient(X, y, theta)
        theta -= lr * grad
        cost = compute_mse_cost(X, y, theta)
        cost_history.append(cost)

        if abs(prev_cost - cost) < tol:
            break

        prev_cost = cost

    elapsed = time.perf_counter() - start

    return theta, cost_history, elapsed


def stochastic_gradient_descent(
    X, y, lr=0.01, epochs=50, tol=1e-7,
    clip_norm=5.0, lr_decay=0.98
):
    m, n_feat = X.shape
    theta = np.zeros((n_feat, 1))
    cost_history = []
    prev_cost = np.inf
    current_lr = lr

    start = time.perf_counter()

    for epoch in range(epochs):
        perm = np.random.permutation(m)

        for i in perm:
            xi = X[i:i + 1]
            yi = y[i:i + 1]
            grad = compute_gradient(xi, yi, theta)

            grad_norm = np.linalg.norm(grad)

            if grad_norm > clip_norm:
                grad = grad * (clip_norm / grad_norm)

            theta -= current_lr * grad

        current_lr *= lr_decay

        cost = compute_mse_cost(X, y, theta)
        cost_history.append(cost)

        if abs(prev_cost - cost) < tol:
            break

        prev_cost = cost

    elapsed = time.perf_counter() - start

    return theta, cost_history, elapsed


def minibatch_gradient_descent(
    X, y, lr=0.05, epochs=100, batch_size=64, tol=1e-7
):
    m, n_feat = X.shape
    theta = np.zeros((n_feat, 1))
    cost_history = []
    prev_cost = np.inf

    start = time.perf_counter()

    for epoch in range(epochs):
        perm = np.random.permutation(m)
        X_shuf, y_shuf = X[perm], y[perm]

        for start_idx in range(0, m, batch_size):
            end_idx = start_idx + batch_size
            X_batch = X_shuf[start_idx:end_idx]
            y_batch = y_shuf[start_idx:end_idx]

            grad = compute_gradient(X_batch, y_batch, theta)
            theta -= lr * grad

        cost = compute_mse_cost(X, y, theta)
        cost_history.append(cost)

        if abs(prev_cost - cost) < tol:
            break

        prev_cost = cost

    elapsed = time.perf_counter() - start

    return theta, cost_history, elapsed


results = {}

theta_bgd, hist_bgd, t_bgd = batch_gradient_descent(
    X_train_b, y_train, lr=0.1, epochs=1000
)
results["Batch GD"] = (theta_bgd, hist_bgd, t_bgd)

theta_sgd, hist_sgd, t_sgd = stochastic_gradient_descent(
    X_train_b, y_train, lr=0.01, epochs=50,
    clip_norm=2.0, lr_decay=0.97
)
results["Stochastic GD"] = (theta_sgd, hist_sgd, t_sgd)

theta_mbgd, hist_mbgd, t_mbgd = minibatch_gradient_descent(
    X_train_b, y_train, lr=0.01, epochs=200, batch_size=64
)
results["Mini-Batch GD"] = (theta_mbgd, hist_mbgd, t_mbgd)

summary_rows = []

for name, (theta, hist, elapsed) in results.items():
    y_pred_test = predict(X_test_b, theta)
    mse_test = compute_mse_cost(X_test_b, y_test, theta)
    r2_test = r2_score(y_test, y_pred_test)
    n_epochs_run = len(hist)
    final_train_cost = hist[-1]

    summary_rows.append({
        "Optimizer": name,
        "Test MSE": mse_test,
        "Test R2": r2_test,
        "Train Time (s)": elapsed,
        "Epochs to Converge": n_epochs_run,
        "Final Train MSE": final_train_cost,
    })

summary_df = pd.DataFrame(summary_rows)
pd.set_option("display.float_format", lambda x: f"{x:0.5f}")

print("\n=== EVALUATION SUMMARY ===")
print(summary_df.to_string(index=False))

coef_df = pd.DataFrame({
    "Feature": ["bias(intercept)"] + FEATURES,
    "Batch GD": theta_bgd.flatten(),
    "Stochastic GD": theta_sgd.flatten(),
    "Mini-Batch GD": theta_mbgd.flatten(),
})

print("\n=== LEARNED COEFFICIENTS (standardized feature space) ===")
print(coef_df.to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, (_, hist, _) in results.items():
    axes[0].plot(hist, label=name)

axes[0].set_title("Convergence: Training MSE per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE (train set)")
axes[0].legend()
axes[0].grid(alpha=0.3)

for name, (_, hist, _) in results.items():
    axes[1].plot(hist, label=name)

axes[1].set_yscale("log")
axes[1].set_title("Convergence: Training MSE per Epoch (log scale)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE (log scale)")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

convergence_path = OUTPUT_DIR / "convergence_plot.png"
plt.savefig(convergence_path, dpi=150)
plt.close(fig)

print(f"\nSaved convergence plot -> {convergence_path}")

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
names = list(results.keys())

axes2[0].bar(
    names,
    [compute_mse_cost(X_test_b, y_test, results[name][0]) for name in names],
    color=["#4C72B0", "#DD8452", "#55A868"]
)
axes2[0].set_title("Test MSE")
axes2[0].tick_params(axis="x", rotation=15)

axes2[1].bar(
    names,
    [
        r2_score(y_test, predict(X_test_b, results[name][0]))
        for name in names
    ],
    color=["#4C72B0", "#DD8452", "#55A868"]
)
axes2[1].set_title("Test R^2 Score")
axes2[1].tick_params(axis="x", rotation=15)

axes2[2].bar(
    names,
    [results[name][2] for name in names],
    color=["#4C72B0", "#DD8452", "#55A868"]
)
axes2[2].set_title("Training Time (s)")
axes2[2].tick_params(axis="x", rotation=15)

plt.tight_layout()

evaluation_path = OUTPUT_DIR / "evaluation_bars.png"
plt.savefig(evaluation_path, dpi=150)
plt.close(fig2)

print(f"Saved evaluation bar chart -> {evaluation_path}")
