# Linear Regression — 2 Features

A **Linear Regression model built from scratch using NumPy and Pandas**, trained with manually implemented Gradient Descent.

The goal of this experiment was to understand the mechanics of Linear Regression and Gradient Descent without relying on `scikit-learn`.

## 📌 Project Overview

This project uses a used-car dataset to predict **selling price** based on two features:

* `year`
* `km_driven`

The model is represented as:

**ŷ = b₀ + b₁x₁ + b₂x₂**

Where:

* `x₁` = year
* `x₂` = km driven
* `b₀` = bias
* `b₁`, `b₂` = learned weights
* `ŷ` = predicted selling price

## 📊 Dataset

The dataset contains information about used cars, including their manufacturing year, kilometers driven, and selling price.

For this experiment, only two features were selected:

| Feature         | Description                   |
| --------------- | ----------------------------- |
| `year`          | Manufacturing year of the car |
| `km_driven`     | Distance driven by the car    |
| `selling_price` | Target variable               |

Using only two features was intentional, allowing the focus to remain on understanding the underlying Linear Regression algorithm.

## ⚙️ Implementation

The model was implemented using:

* **Python**
* **NumPy**
* **Pandas**

No `scikit-learn` was used for training the model.

### Training Process

1. Load the dataset using Pandas.
2. Select `year` and `km_driven` as input features.
3. Select `selling_price` as the target.
4. Standardize the input features.
5. Split the data into training and testing sets.
6. Initialize weights and bias.
7. Calculate predictions.
8. Calculate the prediction error.
9. Calculate gradients.
10. Update the weights and bias using Gradient Descent.
11. Repeat the process for 2,000 epochs.
12. Evaluate the trained model using R² and RMSE.

### Gradient Descent

The gradients were calculated using matrix operations:

* Weight gradient:

`dw = (Xᵀ · error) / n`

* Bias gradient:

`db = mean(error)`

The parameters are then updated using:

`weights = weights - learning_rate × dw`

`bias = bias - learning_rate × db`

## 📈 Results

### Training Set

* **R²:** 0.1747
* **RMSE:** 518,040.41

### Test Set

* **R²:** 0.1613
* **RMSE:** 558,944.27

### Example Prediction

For a car with:

* **Year:** 2015
* **KM Driven:** 40,000

The model predicted a selling price of approximately:

**617,399.89**

## 🔍 Observations

The relatively low R² score shows that using only `year` and `km_driven` does not explain most of the variation in car prices.

Other factors can also influence the selling price, such as:

* Brand
* Fuel type
* Transmission
* Owner history
* Vehicle condition
* Car model

This demonstrates an important Machine Learning concept:

> **Model performance depends not only on the algorithm, but also on the quality and information contained in the features.**

## 🎯 Learning Objectives

This experiment helped me understand and practice:

* Linear Regression
* Multiple features
* Feature standardization
* Train/test splitting
* Matrix operations with NumPy
* Mean Squared Error
* Gradient Descent
* Model parameters
* R²
* RMSE
* Making predictions with a trained model

## 📁 Project Structure

```text
linear-regression-2-features/
│
├── car_data.csv
├── linear_regression.py
└── README.md
```

## 🚀 Future Improvements

Possible extensions of this experiment include:

* Adding more relevant features.
* Comparing performance with different feature combinations.
* Testing different learning rates.
* Comparing Batch, Stochastic, and Mini-Batch Gradient Descent.
* Comparing the from-scratch implementation with `scikit-learn`.
* Experimenting with Polynomial Regression.

---

**Part of my journey toward becoming a Machine Learning Engineer.**
