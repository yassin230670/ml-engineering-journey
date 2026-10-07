import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X = np.linspace(-5, 5, 50).reshape(-1, 1)

y = 2 * X**2 - 3 * X + 5
y = y + np.random.normal(0, 5, size=X.shape)

degree = 2

poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_plot = np.linspace(-5, 5, 200).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)

y_pred = model.predict(X_plot_poly)

plt.scatter(X, y, label="Random Data")

plt.plot(
    X_plot,
    y_pred,
    linewidth=3,
    label=f"Polynomial Regression (degree={degree})"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression")

plt.legend()
plt.grid(True)
plt.show()