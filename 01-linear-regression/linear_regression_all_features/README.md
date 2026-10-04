
# 🚗 Car Price Prediction using Linear Regression

A Machine Learning project that predicts used car prices using **Linear Regression implemented from scratch with NumPy and Pandas**.

The main goal of this project is to understand the fundamental concepts behind regression models, including **feature engineering, data preprocessing, gradient descent, and model evaluation**, without relying on Scikit-learn's Linear Regression implementation.

---

## 📌 Project Overview

In this project, I built a Linear Regression model to predict the `selling_price` of used cars.

The complete workflow includes:

* Loading and exploring the dataset
* Feature engineering
* Categorical feature encoding
* Train/test splitting
* Feature standardization
* Target transformation
* Linear Regression
* Batch Gradient Descent
* Making predictions
* Model evaluation

---

## 📊 Dataset

The project uses a used-car dataset containing information about different vehicles.

The dataset includes features such as:

* Car name
* Manufacturing year
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Previous owner
* Selling price

### Dataset Shape

The dataset contains approximately **4,340 records** and **8 original columns**.

---

## 🔹 Features

The following features were used to train the model:

```text
year
km_driven
fuel
seller_type
transmission
owner
brand
```

### Feature Engineering

The original `name` column was used to extract the car's **brand**:

```python
df["brand"] = df["name"].str.split().str[0]
```

This creates a new feature:

```text
brand
```

---

## 🎯 Target

The target variable is:

```text
selling_price
```

Since car prices can have a large range, the target was transformed using a logarithmic transformation:

```python
y = np.log1p(y)
```

After prediction, the values were converted back to the original price scale:

```python
y_pred = np.expm1(y_pred_log)
```

---

## 🧹 Data Preprocessing

### 1. One-Hot Encoding

Categorical features were converted into numerical features using Pandas:

```python
data = pd.get_dummies(
    data,
    columns=[
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "brand"
    ],
    dtype=float
)
```

### 2. Train/Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

A random seed of `42` was used to make the split reproducible.

### 3. Feature Standardization

The numerical features were standardized using the training-set mean and standard deviation:

```python
X_standardized = (X - mean) / std
```

The two numerical features standardized were:

* `year`
* `km_driven`

Importantly, the statistics were calculated **only from the training data** and then applied to both training and testing data.

---

# 🤖 Linear Regression

The model predicts the target using:

$$
\hat{y} = Xw
$$

Where:

* `X` = input features
* `w` = model weights
* `ŷ` = predicted value

A bias/intercept term was added to the feature matrix:

```python
X_train = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]
```

---

# ⚙️ Gradient Descent

The Linear Regression model was trained using **Batch Gradient Descent**, implemented manually with NumPy.

### Mean Squared Error

The cost function is:

$$
J(w) = \frac{1}{m}\sum_{i=1}^{m}(\hat{y_i}-y_i)^2
$$

The gradient is:

$$
\nabla J(w) = \frac{2}{m}X^T(Xw-y)
$$

The weights are updated using:

$$
w = w - \alpha\nabla J(w)
$$

Where:

* `α` = learning rate
* `m` = number of training examples
* `w` = model weights

### Training Configuration

```text
Algorithm: Linear Regression
Optimizer: Batch Gradient Descent
Learning Rate: 0.01
Epochs: 3000
```

---

# 📈 Model Evaluation

The model was evaluated on the test set using three metrics.

### R² Score

Measures how much of the variance in the target variable is explained by the model.

$$
R^2 = 1 - \frac{SS_{res}}{SS_{total}}
$$

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

$$
MAE = \frac{1}{n}\sum |y-\hat{y}|
$$

### RMSE — Root Mean Squared Error

Penalizes larger errors more heavily.

$$
RMSE = \sqrt{\frac{1}{n}\sum(y-\hat{y})^2}
$$

The metrics were calculated after converting the predictions back to the original price scale.

---

## 🧪 Sample Prediction Output

The project also creates a DataFrame comparing actual and predicted prices:

```python
results = pd.DataFrame({
    "Actual Price": y_actual,
    "Predicted Price": y_pred
})
```

Example:

```text
Actual Price    Predicted Price
-------------   ---------------
450000          421XXX
350000          38XXXX
600000          57XXXX
...
```

---

# 🛠️ Technologies Used

* **Python**
* **NumPy**
* **Pandas**

No Scikit-learn regression model was used.

---

# 📁 Project Structure

```text
car-price-prediction/
│
├── car_data.csv
├── linear_regression.py
├── README.md
└── requirements.txt
```

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/car-price-prediction.git
```

### 2. Navigate to the project

```bash
cd car-price-prediction
```

### 3. Install dependencies

```bash
pip install numpy pandas
```

Or:

```bash
pip install -r requirements.txt
```

### 4. Run the model

```bash
python linear_regression.py
```

---

# 🎯 What I Learned

This project helped me understand the complete workflow of a regression problem and how the underlying mathematics connects to the implementation.

Key concepts practiced:

* Feature engineering
* One-hot encoding
* Train/test splitting
* Data standardization
* Log transformation
* Linear Regression
* Mean Squared Error
* Batch Gradient Descent
* Weight updates
* R²
* MAE
* RMSE
* NumPy matrix operations

---

# 🚀 Future Improvements

Possible improvements for this project:

* Compare different learning rates
* Compare Batch GD, SGD, and Mini-Batch GD
* Implement Momentum
* Experiment with Polynomial Regression
* Add visualizations for training loss
* Compare different feature combinations
* Experiment with additional regression algorithms
* Perform cross-validation

---

## 👨‍💻 Author

**Yassin Mohamed**

Computer Science | Machine Learning & AI

This project is part of my journey toward becoming a **Machine Learning Engineer**.

---

⭐ If you found this project useful, consider giving the repository a star!
