# Linear Regression from Scratch: Comparing Gradient Descent Algorithms

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-from%20scratch-013243?logo=numpy&logoColor=white)
![Status](https://img.shields.io/badge/status-completed-brightgreen)

As part of my journey toward becoming a **Machine Learning Engineer**, I implemented Linear Regression from scratch and compared three Gradient Descent variations on the California Housing dataset.

The goal was to understand how different optimization strategies affect model performance, convergence, and training time, without relying on pre-built Linear Regression implementations.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Gradient Descent Algorithms](#gradient-descent-algorithms)
- [Core Mathematical Concepts](#core-mathematical-concepts)
- [Experimental Results](#experimental-results)
- [Technologies Used](#technologies-used)
- [How to Run the Project](#how-to-run-the-project)
- [Project Structure](#project-structure)
- [Key Learnings](#key-learnings)
- [What's Next](#whats-next)

## Project Overview

This experiment compares three optimization algorithms:

- **Batch Gradient Descent (BGD)**
- **Stochastic Gradient Descent (SGD)**
- **Mini-Batch Gradient Descent (MBGD)**

The model's predictions, Mean Squared Error (MSE), gradients, and parameter updates are all implemented manually using NumPy.

## Dataset

The experiment uses the California Housing dataset.

| Property | Value |
|---|---|
| Total records | 20,640 |
| Input features | 8 |
| Training set | 16,512 records (80%) |
| Test set | 4,128 records (20%) |
| Target | Median house value (`MedHouseVal`) |

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average rooms per household |
| `AveBedrms` | Average bedrooms per household |
| `Population` | Block-group population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

## Data Preprocessing

1. Removed missing values.
2. Split the dataset into training and testing sets.
3. Capped extreme feature values using the training set's 1st and 99th percentiles.
4. Standardized features using training-set statistics.
5. Added a bias column to represent the intercept.

> [!NOTE]
> All preprocessing statistics are computed from the training data only, to avoid data leakage from the test set.

## Gradient Descent Algorithms

### 1. Batch Gradient Descent (BGD)

Calculates the gradient using the **entire training dataset** before each parameter update.

- **Advantages:** Stable gradient estimates and efficient full-batch matrix operations.
- **Limitation:** Each update processes the entire dataset.

### 2. Stochastic Gradient Descent (SGD)

Calculates the gradient using **one training example** at a time.

- **Advantages:** Frequent parameter updates and low per-update computational cost.
- **Limitations:** Noisier updates and potentially greater sensitivity to learning rate and outliers.

### 3. Mini-Batch Gradient Descent (MBGD)

Calculates the gradient using a **small batch** of training examples.

- **Advantages:** Balances gradient stability with computational efficiency and supports vectorized operations.
- **Limitation:** Performance depends on batch size and optimization settings.

## Core Mathematical Concepts

### Linear Regression

The model predicts the target using a weighted sum of its features:

$$
\hat{y} = X\theta
$$

Where:

- $X$ is the feature matrix with an added bias column.
- $\theta$ contains the weights and the intercept.
- $\hat{y}$ represents the predicted values.

### Mean Squared Error

The cost function measures the average squared prediction error:

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

### Gradient Descent Update

The parameters are updated in the direction that reduces the cost:

$$
\theta := \theta - \alpha \nabla_\theta J(\theta)
$$

Where $\alpha$ is the learning rate and $\nabla_\theta J(\theta)$ is the gradient of the cost function.

The main difference between BGD, SGD, and MBGD is the **number of training examples used to calculate each gradient update**.

## Experimental Results

All three models were evaluated on the same test set.

| Optimizer | Test MSE ↓ | Test R² ↑ | Training Time |
|---|---:|---:|---:|
| Batch GD | **0.46292** | **0.65607** | **0.04090 s** |
| Stochastic GD | 0.47347 | 0.64823 | 1.94399 s |
| Mini-Batch GD | 0.46328 | 0.65580 | 0.14481 s |

### Results Analysis

**Batch Gradient Descent**

- Achieved the lowest Test MSE and highest R² score.
- Had the shortest measured training time in this implementation.
- Produced stable convergence on this dataset.

**Stochastic Gradient Descent**

- Had slightly higher Test MSE and lower R² than the other methods.
- Took the longest to train in this implementation.
- Used gradient clipping and learning-rate decay to control optimization updates.

**Mini-Batch Gradient Descent**

- Achieved results very close to Batch GD.
- Offered a competitive balance between prediction performance and gradient-update size.
- Could be particularly useful for larger datasets and hardware-accelerated training.

> [!IMPORTANT]
> These results are specific to the dataset, learning rates, stopping criteria, and implementation used. They do not establish that one optimizer is universally superior.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- `pathlib` for cross-platform file and directory management

The model is implemented from scratch **without** using `sklearn.linear_model` or automatic differentiation.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/linear-regression-gradient-descent-comparison.git
cd linear-regression-gradient-descent-comparison
```

### 2. Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install numpy pandas matplotlib
```

Or, if you have a `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Place `california_housing.csv` in the same directory as the Python script.

The CSV must contain the eight feature columns and the `MedHouseVal` target column listed above.

### 5. Run the experiment

```bash
python california_housing.py
```

The evaluation summary and learned coefficients are printed in the terminal, and the output plots are saved in the `outputs/` directory.

## Project Structure

```text
linear-regression-gradient-descent-comparison/
├── california_housing.py
├── california_housing.csv
├── outputs/
│   ├── convergence_plot.png
│   └── evaluation_bars.png
├── requirements.txt
└── README.md
```

> [!TIP]
> The dataset and generated images may be excluded from version control depending on repository size and licensing considerations.

## Key Learnings

Through this experiment, I strengthened my understanding of:

- Implementing Linear Regression mathematically from scratch.
- Calculating gradients and updating model parameters.
- Understanding the differences between BGD, SGD, and MBGD.
- Applying feature scaling and outlier handling.
- Evaluating regression models using MSE and R².
- Comparing convergence behavior and training time.
- Organizing Python projects using portable file paths.

## What's Next

This experiment is one step in my ongoing journey toward becoming a Machine Learning Engineer. Next, I plan to:

- Explore more advanced optimization techniques.
- Investigate regularization.
- Move toward classification algorithms.


---

*Learning, experimenting, and building — one step at a time.*
