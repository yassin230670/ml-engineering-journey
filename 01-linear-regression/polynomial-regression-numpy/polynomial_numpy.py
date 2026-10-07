import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.linspace(-5, 5, 50)

y = 2 * X**2 - 3 * X + 5
y = y + np.random.normal(0, 5, size=len(X))

degree = 2

X_poly = np.column_stack([
    X**i for i in range(degree + 1)
])

theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

X_plot = np.linspace(-5, 5, 200)

X_plot_poly = np.column_stack([
    X_plot**i for i in range(degree + 1)
])

y_pred = X_plot_poly @ theta

plt.scatter(X, y, label="Random Data")

plt.plot(
    X_plot,
    y_pred,
    linewidth=3,
    label=f"Polynomial Regression (degree={degree})"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression using NumPy")

plt.legend()
plt.grid(True)
plt.show()