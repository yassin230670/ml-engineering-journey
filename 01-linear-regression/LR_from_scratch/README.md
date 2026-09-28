
# Linear Regression From Scratch

A from-scratch implementation of **Linear Regression using Python and NumPy**, built as part of my journey to becoming a Machine Learning Engineer.

The goal of this project is to understand how Linear Regression works internally rather than relying on pre-built machine learning libraries.

---

## 📌 Overview

Linear Regression is a supervised learning algorithm used to model the relationship between input features and a continuous numerical target.

For a simple Linear Regression model:

\[
\hat{y} = wx + b
\]

Where:

- \(x\) → input feature
- \(\hat{y}\) → predicted value
- \(w\) → weight / slope
- \(b\) → bias / intercept

In this project, the model learns `w` and `b` using **Batch Gradient Descent**.

---

## 🎯 Project Example

The model uses a simple dataset representing the relationship between:

**Study Hours → Exam Score**

| Study Hours | Exam Score |
| ----------: | ---------: |
|           2 |         50 |
|           4 |         65 |
|           6 |         78 |
|           8 |         90 |
|          10 |         98 |

The goal is to learn a relationship that allows the model to predict the expected score based on the number of study hours.

For example:

```text
Study Hours: 7
        ↓
Linear Regression Model
        ↓
Predicted Score
```
