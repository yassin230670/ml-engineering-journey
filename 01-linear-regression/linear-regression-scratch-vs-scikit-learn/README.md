# Linear Regression: From Scratch vs Scikit-Learn 🚗

A practical Machine Learning experiment comparing **Linear Regression implemented from scratch using NumPy** with **Scikit-Learn's LinearRegression** for car price prediction.

This project is part of my journey to becoming a **Machine Learning Engineer**.

---

## 📌 Project Overview

The goal of this experiment was to understand how Linear Regression works internally and compare my own implementation with a production-ready Machine Learning library.

I used a car dataset containing **4,340 records** and predicted car prices based on features such as:

* Car brand
* Year
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Owner

Two models were trained using the same processed data:

1. **Linear Regression from Scratch**
2. **Scikit-Learn Linear Regression**

---

## 🔄 Machine Learning Pipeline

```text
Raw Dataset
     ↓
Feature Engineering
     ↓
Brand Extraction
     ↓
One-Hot Encoding
     ↓
Log Transformation
     ↓
Train/Test Split
     ↓
Feature Standardization
     ↓
Model Training
     ↓
Predictions
     ↓
Model Evaluation
```

---

## 🛠️ Data Preprocessing

### 1. Brand Extraction

The original `name` column contains the full car name.

For example:

```text
Maruti 800 AC → Maruti
Hyundai Verna 1.6 SX → Hyundai
Honda Amaze VX i-DTEC → Honda
```

The first word was extracted and used as the car's **brand**.

### 2. One-Hot Encoding

Categorical features were converted into numerical features using One-Hot Encoding.

Encoded features include:

* `fuel`
* `seller_type`
* `transmission`
* `owner`
* `brand`

### 3. Target Transformation

Because car prices have a wide range, the target variable was transformed using:

```python
y = np.log1p(y)
```

After prediction, the transformation was reversed using:

```python
y_pred = np.expm1(y_pred_log)
```

### 4. Feature Standardization

The numerical features:

* `year`
* `km_driven`

were standardized using the mean and standard deviation calculated from the training data.

---

## 🧠 Linear Regression From Scratch

The model was implemented using **NumPy**, without using Scikit-Learn for training.

### Prediction

```text
ŷ = Xw
```

### Error

```text
error = ŷ - y
```

### Mean Squared Error

```text
MSE = mean(error²)
```

### Gradient

```text
gradient = (2/m) Xᵀ(error)
```

### Weight Update

```text
w = w - α × gradient
```

The model was trained using:

* Learning rate: `0.01`
* Epochs: `3000`

The training cost decreased from approximately:

```text
163.635 → 0.206
```

---

## 🤖 Scikit-Learn Model

The second model used:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

The same training and testing data were used to make the comparison fair.

---

## 📊 Results

| Metric   | From Scratch |   Scikit-Learn |
| -------- | -----------: | -------------: |
| **R²**   |       0.6072 |     **0.7970** |
| **MAE**  |   171,386.48 | **138,596.30** |
| **RMSE** |   382,523.22 | **275,026.92** |

### Interpretation

Scikit-Learn achieved better results in this experiment.

The important part, however, was understanding **why**.

My implementation uses **iterative Gradient Descent** to optimize the weights.

Scikit-Learn's `LinearRegression` uses a **numerical linear-algebra solution** to solve the ordinary least-squares problem.

Therefore, both approaches solve the same underlying Linear Regression problem, but they use different methods to obtain the coefficients.

---

## 📚 What I Learned

This experiment helped me understand several important concepts:

* How to preprocess real-world tabular data
* Feature engineering
* One-Hot Encoding
* Target transformations
* Feature standardization
* Mean Squared Error
* Gradient Descent
* Vectorized NumPy operations
* Model evaluation
* R², MAE, and RMSE
* The difference between iterative optimization and numerical solutions
* What happens underneath a Machine Learning library

Most importantly, implementing the algorithm from scratch helped me understand what happens behind:

```python
model.fit(X, y)
```

rather than treating the library as a black box.

---

## 📁 Project Structure

```text
linear-regression-scratch-vs-scikit-learn/
│
├── car_data.csv
├── linearRegresionComparison.py
└── README.md
```

---

## ⚙️ Requirements

Install the required libraries with:

```bash
pip install numpy pandas scikit-learn
```

---

## ▶️ How to Run

Clone the repository:

```bash
git clone <your-repository-url>
```

Navigate to the project folder:

```bash
cd linear-regression-scratch-vs-scikit-learn
```

Run the Python script:

```bash
python linearRegresionComparison.py
```

---

## 🚀 Part of My ML Journey

This project is part of my ongoing series:

**"My Journey to Becoming a Machine Learning Engineer"**

The goal is to learn Machine Learning by building projects, implementing algorithms from scratch, and understanding the mathematics and concepts behind the tools I use.

### 🔗 Connect With Me

* **GitHub:** [[Your GitHub]](https://github.com/yassin230670)
* **Instagram:** https://www.instagram.com/yassin_log/
* **TikTok:** https://www.tiktok.com/@yassin.cs?lang=en-GB
* **YouTube:** https://www.youtube.com/@yassincs-5885

---

⭐ If you found this project useful, feel free to explore the repository and follow my Machine Learning journey.

