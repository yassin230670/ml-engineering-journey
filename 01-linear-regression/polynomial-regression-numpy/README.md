# Polynomial Regression with NumPy

This project is part of my **Journey to Becoming a Machine Learning Engineer**.

The goal is to understand how **Polynomial Regression** works and how to implement it from scratch using **NumPy**, without relying on Scikit-learn for the regression model.

## 📌 What is Polynomial Regression?

Polynomial Regression is an extension of Linear Regression that can model **non-linear relationships** between the input and target.

Instead of using only:

$$
y = \beta_0 + \beta_1x
$$

Polynomial Regression can use higher-order terms:

$$
y = \beta_0 + \beta_1x + \beta_2x^2 + \cdots + \beta_nx^n
$$

The polynomial degree controls how flexible the model can be.

## 🧠 What I Implemented

* Generated random data using NumPy
* Added random noise to the data
* Created polynomial features
* Implemented Polynomial Regression using NumPy
* Calculated the coefficients using the **Normal Equation**
* Generated predictions
* Visualized the fitted polynomial curve using Matplotlib
* Experimented with different polynomial degrees

## 🛠️ Technologies

* Python
* NumPy
* Matplotlib

## 📊 Example

The model learns a polynomial relationship from randomly generated data and fits a curve through the observations.

You can change:

```python
degree = 2
```

to experiment with different polynomial degrees.

For example:

* `degree = 1` → Linear Regression
* `degree = 2` → Quadratic Regression
* `degree = 3` → Cubic Regression
* `degree = 4` → Fourth-degree Polynomial Regression

## 📁 Project Structure

```text
polynomial-regression-numpy/
│
├── polynomial_regression.py
└── README.md
```

## 🚀 Running the Project

Install the required libraries:

```bash
pip install numpy matplotlib
```

Then run:

```bash
python polynomial_regression.py
```

## 🎯 Learning Goal

This project is mainly focused on understanding the mathematical and practical idea behind Polynomial Regression rather than simply using a machine learning library.

It is one step in my ongoing journey to become a **Machine Learning Engineer**.

## 🔗 Connect With Me



---

**Part of my Machine Learning Journey 🚀**
GitHub: https://github.com/yassin230670
Instagram: https://www.instagram.com/yassin_log/
TikTok: https://www.tiktok.com/@yassin.cs?lang=en-GB
YouTube: https://www.youtube.com/@yassincs-5885
