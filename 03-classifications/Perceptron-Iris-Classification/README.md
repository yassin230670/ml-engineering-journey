# Perceptron Iris Classification — From Scratch with NumPy

A Machine Learning project implementing the **Perceptron algorithm from scratch using NumPy** and applying it to the Iris dataset.

The project explores binary and multiclass classification through three experiments, evaluates model performance, and visualizes the learning process using decision boundaries, learning curves, and confusion matrices.

## Overview

The goal of this project is to understand how the Perceptron learning algorithm works internally and investigate its ability to classify different Iris flower species.

Instead of relying on a ready-made Perceptron implementation, the core learning algorithm is implemented using NumPy.

The project covers the complete Machine Learning workflow:

- Data loading and exploration.
- Data preprocessing and feature scaling.
- Perceptron implementation from scratch.
- Binary and multiclass classification.
- Model evaluation and performance analysis.
- Learning visualization and experiment comparison.

## Dataset

This project uses the **Iris Dataset**, a classic dataset in Machine Learning containing measurements of three Iris flower species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

The dataset contains 150 samples, with 50 samples per species.

### Features

| Feature | Description |
|---|---|
| SepalLengthCm | Sepal length in centimeters |
| SepalWidthCm | Sepal width in centimeters |
| PetalLengthCm | Petal length in centimeters |
| PetalWidthCm | Petal width in centimeters |
| Species | Target class |

The `Id` column is removed because it is an identifier rather than a meaningful predictive feature.

## Experiments

Three classification experiments are conducted.

### Experiment 1: Setosa vs. Versicolor

A binary classification task where the Perceptron learns to distinguish between:

- Iris Setosa (Class 0)
- Iris Versicolor (Class 1)

### Experiment 2: Versicolor vs. Virginica

A second binary classification task using:

- Iris Versicolor (Class 0)
- Iris Virginica (Class 1)

This experiment investigates how the algorithm performs when distinguishing a different pair of species.

### Experiment 3: Multiclass Classification — One-vs-Rest

The One-vs-Rest (OvR) strategy extends binary classification to three classes.

Three independent binary Perceptron classifiers are trained:

1. Setosa vs. Rest
2. Versicolor vs. Rest
3. Virginica vs. Rest

The final predicted class is selected by comparing the classifiers' scores.

## Perceptron Algorithm

The Perceptron is a supervised learning algorithm used for binary classification.

### 1. Linear combination

The model calculates a weighted sum of the input features:

\[
z = \mathbf{w}^{T}\mathbf{x} + b
\]

Where:

- \( \mathbf{x} \): Input feature vector.
- \( \mathbf{w} \): Weight vector.
- \( b \): Bias.
- \( z \): Model output before activation.

### 2. Step activation function

The predicted class is determined using a step function:

\[
\hat{y} =
\begin{cases}
1 & z \geq 0 \\
0 & z < 0
\end{cases}
\]

### 3. Parameter updates

The algorithm updates its weights and bias when a prediction is incorrect:

\[
\mathbf{w} \leftarrow
\mathbf{w} + \eta(y-\hat{y})\mathbf{x}
\]

\[
b \leftarrow b + \eta(y-\hat{y})
\]

Where:

- \(y\): Actual label.
- \(\hat{y}\): Predicted label.
- \(\eta\): Learning rate.

The training process repeats over multiple epochs until the stopping condition is reached.

## Project Structure

```text
perceptron-iris-classification/
│
├── data/
│   └── Iris.csv
│
├── src/
│   ├── perceptron.py
│   ├── data_preprocessing.py
│   ├── experiments.py
│   ├── evaluation.py
│   └── visualization.py
│
├── results/
│
├── tests/
│   └── test_perceptron.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `perceptron.py` | Implements the Perceptron algorithm and training process. |
| `data_preprocessing.py` | Loads, validates, filters, splits, and scales the dataset. |
| `experiments.py` | Defines and executes the three classification experiments. |
| `evaluation.py` | Calculates classification metrics and evaluation results. |
| `visualization.py` | Generates learning curves, decision plots, and comparison visualizations. |
| `main.py` | Entry point for running the complete project. |
| `test_perceptron.py` | Tests the core Perceptron implementation. |
| `results/` | Stores generated plots and experiment outputs. |

## Data Preprocessing

The preprocessing pipeline includes:

1. Loading the Iris CSV file using Pandas.
2. Removing the unnecessary `Id` column.
3. Selecting the four numerical features.
4. Filtering the dataset according to each experiment.
5. Encoding species labels into numerical classes for binary classification.
6. Splitting the data into training and testing sets.
7. Standardizing features using `StandardScaler`.

A fixed random seed is used to support reproducibility.

**Note:** The scaler is fitted on training data and then applied to the test data to prevent data leakage.

## Evaluation Metrics

The experiments use classification metrics to assess model performance:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive samples correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications for each class.

## Visualizations

The project generates visualizations to investigate the learning process and classification behavior.

Depending on the experiment, the generated plots include:

- Binary classification decision boundaries.
- Training learning curves.
- Number of misclassifications per epoch.
- Multiclass confusion matrix.
- Multiclass decision regions.
- Experiment performance comparison.

All generated figures are saved in the `results/` directory.

## Technologies & Libraries

- **Python** — Main programming language.
- **NumPy** — Perceptron implementation and numerical operations.
- **Pandas** — Dataset loading and manipulation.
- **Matplotlib** — Data visualization.
- **Scikit-learn** — Data splitting, feature scaling, and evaluation utilities where used.
- **Pytest** — Automated testing.

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/perceptron-iris-classification.git

cd perceptron-iris-classification
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place `Iris.csv` inside the `data/` directory.

### 5. Run the project

```bash
python main.py
```

The project executes the three experiments, prints evaluation results in the terminal, and saves the generated visualizations in the `results/` directory.

### 6. Run tests

```bash
python -m pytest
```

## Results & Performance

The project evaluates the Perceptron across three classification tasks.

The following table can be populated with the actual test-set results after running the experiments:

| Experiment | Classification Type | Test Accuracy |
|---|---|---|
| Setosa vs. Versicolor | Binary | Add result |
| Versicolor vs. Virginica | Binary | Add result |
| One-vs-Rest | Multiclass | Add result |

The results can be used to investigate:

- How classification difficulty affects model performance.
- How the Perceptron learns a linear decision boundary.
- How One-vs-Rest extends a binary classifier to multiclass classification.
- How training errors change across epochs.
- The limitations of linear classification.

## Key Learning Outcomes

Through this project, I explored:

- Implementing a Machine Learning algorithm from scratch.
- Understanding the mathematical foundations of the Perceptron.
- Working with NumPy for numerical computation.
- Building reusable and modular Python code.
- Designing binary and multiclass classification experiments.
- Applying data preprocessing and feature scaling.
- Evaluating models using multiple classification metrics.
- Visualizing model behavior and interpreting experimental results.

## Future Improvements

Potential extensions include:

- Comparing the implementation with Scikit-learn's Perceptron.
- Experimenting with different learning rates and initialization methods.
- Testing different feature combinations.
- Applying cross-validation.
- Comparing Perceptron with Logistic Regression and other classifiers.
- Investigating non-linearly separable datasets.

## Author

**Yassin**

Computer Science Graduate | Machine Learning Enthusiast

This project is part of my journey toward becoming a Machine Learning Engineer, documenting my practical learning and experiments with Machine Learning algorithms.

---

**If you find this project useful, feel free to star the repository!**
