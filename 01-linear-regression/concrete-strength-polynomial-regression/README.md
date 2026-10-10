# Concrete Compressive Strength Prediction — Polynomial Regression

## 📌 Overview

As part of my journey toward becoming an **AI Engineer**, I conducted a Machine Learning experiment to predict the compressive strength of concrete using its composition and curing age.

The goal of this project is to investigate how polynomial feature engineering affects Linear Regression performance and to understand the relationship between model complexity and generalization.

Using the Concrete Compressive Strength dataset, I trained and compared four regression models with different polynomial degrees.

## 🎯 Objectives

- Explore and analyze a real-world regression dataset.
- Perform Exploratory Data Analysis (EDA).
- Apply polynomial feature engineering to capture nonlinear relationships.
- Compare polynomial degrees from 1 to 4.
- Evaluate models using multiple regression metrics.
- Use 5-fold cross-validation to help select the best polynomial degree.
- Investigate potential underfitting and overfitting.

## 📊 Dataset

**Dataset:** Concrete Compressive Strength

The dataset contains measurements of concrete ingredients and curing age, with compressive strength as the target variable.

### Input Features

| Feature | Description |
|---|---|
| Cement | Cement quantity |
| Blast Furnace Slag | Slag quantity |
| Fly Ash | Fly ash quantity |
| Water | Water quantity |
| Superplasticizer | Superplasticizer quantity |
| Coarse Aggregate | Coarse aggregate quantity |
| Fine Aggregate | Fine aggregate quantity |
| Age | Concrete curing age in days |

**Target variable:** Concrete compressive strength, measured in MPa.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## ⚙️ Project Workflow

### 1. Data Loading and Cleaning

- Load the dataset from an Excel file.
- Rename columns for easier analysis.
- Inspect missing values and duplicate records.
- Remove duplicate rows.

### 2. Exploratory Data Analysis (EDA)

Investigate the dataset using:

- Statistical summaries.
- Feature distribution histograms.
- Correlation heatmap.
- Boxplots for potential outlier detection.

### 3. Train-Test Split

Split the dataset into:

- **80% training data**
- **20% testing data**

The experiment uses `random_state=42` to make the split reproducible.

### 4. Polynomial Feature Engineering

Apply `PolynomialFeatures` with degrees 1, 2, 3, and 4.

Polynomial expansion introduces higher-order terms and feature interactions, allowing a linear regression algorithm to model more complex relationships.

For example, with two input variables, a degree-2 expansion can include:

\[
x_1,\;x_2,\;x_1^2,\;x_1x_2,\;x_2^2
\]

The resulting model is linear in its learned coefficients, even though it can represent nonlinear relationships in the original input features.

### 5. Feature Scaling

Apply `StandardScaler` after polynomial expansion.

The scaler is fitted on the training data and then used to transform both training and testing data, preventing the test set from influencing the scaling parameters.

### 6. Model Training

Train a separate `LinearRegression` model for each polynomial degree:

- Degree 1 — Linear Regression
- Degree 2 — Quadratic Polynomial Regression
- Degree 3 — Cubic Polynomial Regression
- Degree 4 — Fourth-degree Polynomial Regression

### 7. Model Evaluation

Evaluate each model using the following metrics:

| Metric | Purpose |
|---|---|
| R² Score | Measures how much target variance is explained by the model |
| RMSE | Measures prediction error while penalizing larger errors more heavily |
| MAE | Measures average absolute prediction error |
| 5-Fold CV R² | Evaluates performance across multiple validation folds |

The experiment compares training performance, test performance, and cross-validation scores to help assess model complexity and generalization.

### 8. Model Selection

Select the polynomial degree with the highest mean cross-validation R² score, then evaluate the selected model on the held-out test set.

**Note:** The best degree and final performance metrics depend on the results obtained when running the experiment.

## 📈 Visualizations

The script generates the following plots:

| Visualization | Description |
|---|---|
| `distributions.png` | Feature and target distributions |
| `correlation_heatmap.png` | Correlations between numerical variables |
| `boxplots.png` | Potential outliers in input features |
| `degree_comparison.png` | Training, testing, and cross-validation R² across polynomial degrees |
| `predicted_vs_actual.png` | Comparison of actual and predicted concrete strength |

These visualizations help analyze the data, compare model complexity, and assess prediction quality.

## 📂 Project Structure

```text
concrete-strength-polynomial-regression/
│
├── Concrete_Data.xls
├── main.py
├── distributions.png
├── correlation_heatmap.png
├── boxplots.png
├── degree_comparison.png
├── predicted_vs_actual.png
├── requirements.txt
└── README.md
```

*The image files are generated when the script runs. `main.py` represents the Python script containing the experiment.*

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd concrete-strength-polynomial-regression
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xlrd
openpyxl
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run the Experiment

Ensure `Concrete_Data.xls` is in the project directory, then execute:

```bash
python main.py
```

The program prints the evaluation metrics and saves the generated visualizations in the current directory.

## 🧠 Key Learning Outcomes

Through this experiment, I explored:

- How polynomial feature engineering expands the representation of input data.
- How model complexity can affect training and testing performance.
- Why evaluating a model using multiple metrics provides a more complete picture.
- How cross-validation can support model selection.
- Why a more complex model is not necessarily a better model.
- How to build a structured regression workflow using Scikit-learn.

## 🔍 Future Improvements

- Use a Scikit-learn `Pipeline` to perform preprocessing correctly within each cross-validation fold.
- Investigate Ridge and Lasso regularization to control model complexity.
- Compare Polynomial Regression with other regression algorithms.
- Perform systematic hyperparameter tuning.
- Add residual analysis and learning curves.
- Document the final experimental results in a comparison table.

## 👨‍💻 About This Project

This project is part of my ongoing journey toward becoming an **AI Engineer**, where I document my experiments, strengthen my Machine Learning foundations, and develop practical skills through implementation and evaluation.

I believe the best way to learn Machine Learning is to experiment, analyze results, and understand not only what works, but why.

**More experiments and projects will follow.**
