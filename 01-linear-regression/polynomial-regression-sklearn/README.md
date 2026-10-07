# Polynomial Regression with Scikit-Learn

This project is part of my **Journey to Becoming a Machine Learning Engineer**.

The goal is to understand how Polynomial Regression works by transforming the original features into polynomial features and then applying Linear Regression using Scikit-Learn.

## 📌 What is Polynomial Regression?

Polynomial Regression is a technique used to model a **non-linear relationship** between the input and target variable.

Instead of using only the original feature:

$$
X
$$

we transform it into polynomial features:

$$
[1, X, X^2, X^3, \dots, X^n]
$$

Then, we apply Linear Regression to these transformed features.

For example, with degree 2:

$$
\hat{y} = \beta_0 + \beta_1X + \beta_2X^2
$$

Although the resulting relationship is a curve, the model is still a **Linear Regression model with respect to its parameters**.

## 🧠 What I Implemented

The project follows these steps:

1. Generate random `(X, y)` data using NumPy.
2. Add random noise to the target values.
3. Transform the input using `PolynomialFeatures`.
4. Apply `LinearRegression` from Scikit-Learn.
5. Generate predictions.
6. Visualize the original data and the polynomial regression curve using Matplotlib.

## 🛠️ Technologies

* Python
* NumPy
* Matplotlib
* Scikit-Learn

## 📊 Polynomial Degree

The polynomial degree determines how many polynomial features are created.

For example:

```python
degree = 1
```

produces:

$$
[1, X]
$$

```python
degree = 2
```

produces:

$$
[1, X, X^2]
$$

```python
degree = 3
```

produces:

$$
[1, X, X^2, X^3]
$$

Higher degrees allow the model to represent more complex relationships, but they can also increase the risk of **overfitting**.

## 🔄 Workflow

```text
Random Data
     ↓
PolynomialFeatures
     ↓
Polynomial Features
     ↓
LinearRegression
     ↓
Predictions
     ↓
Visualization
```

## 📁 Project Structure

```text
polynomial-regression-sklearn/
│
├── polynomial_regression.py
└── README.md
```

## 🚀 How to Run

Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn
```

Then run:

```bash
python polynomial_regression.py
```

## 🎯 Learning Objective

This project helped me understand an important concept in Machine Learning:

> **Polynomial Regression can be implemented by transforming the features into polynomial terms and then applying Linear Regression.**

This is another step in my journey toward becoming a **Machine Learning Engineer**.

## 🔗 Connect With Me

* GitHub: https://github.com/yassin230670
* Instagram: https://www.instagram.com/yassin_log/
* TikTok: https://www.tiktok.com/@yassin.cs?lang=en-GB
* YouTube: https://www.youtube.com/@yassincs-5885

---

**Part of my Machine Learning Journey 🚀**
