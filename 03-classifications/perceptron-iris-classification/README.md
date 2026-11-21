# Perceptron From Scratch — Iris Classification

A NumPy implementation of the classic online Perceptron with three Iris experiments:
1. Setosa vs. Versicolor
2. Versicolor vs. Virginica
3. One-vs-Rest classification across all three species

## Requirements

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
python -m pip install -r requirements.txt
```

Place the Iris CSV at `data/Iris.csv`. Expected columns are `Id` (optional), `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, and `Species`.

## Run

```bash
python main.py
```

Or provide another dataset path:

```bash
python main.py --data /path/to/Iris.csv
```

Generated plots and comparison reports are saved to `results/`.

## Algorithm

For binary labels 0 and 1, the model calculates `z = w @ x + b`, predicts 1 when `z >= 0`, and otherwise predicts 0. For a misclassified sample, it updates `w = w + learning_rate * (y - y_hat) * x` and `b = b + learning_rate * (y - y_hat)`.

The model trains online, updating one sample at a time. The convergence flag means an epoch had zero training mistakes. It does not imply good test performance.

## Experiments

The first two experiments use all four features for evaluation and separate two-feature models for visualizing petal-length/petal-width decision boundaries. The third experiment trains three binary models using One-vs-Rest. It chooses the class with the largest raw linear score. These scores are not probabilities and may have different scales.

## Tests

```bash
pytest -q
```

## Limitations

A classic Perceptron can converge to zero training mistakes when the training data are linearly separable under the algorithm's assumptions. It is not guaranteed to converge on non-linearly-separable data. The Iris dataset is small, so the reported split results are illustrative rather than a definitive benchmark.
