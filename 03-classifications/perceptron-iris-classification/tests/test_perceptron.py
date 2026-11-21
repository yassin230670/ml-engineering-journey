import numpy as np
import pytest
from src.perceptron import Perceptron


def test_perceptron_learns_simple_separable_data():
    X = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = Perceptron(learning_rate=0.1, max_epochs=100, random_state=7)
    model.fit(X, y)
    assert np.mean(model.predict(X) == y) == 1.0
    assert model.converged_


def test_predict_shape():
    X = np.array([[-1.0], [1.0]])
    y = np.array([0, 1])
    model = Perceptron(max_epochs=20).fit(X, y)
    assert model.predict(X).shape == (2,)


def test_invalid_labels_raise():
    with pytest.raises(ValueError):
        Perceptron().fit(np.array([[0.], [1.]]), np.array([0, 2]))


def test_invalid_feature_values_raise():
    with pytest.raises(ValueError):
        Perceptron().fit(np.array([[np.nan], [1.]]), np.array([0, 1]))
