from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

COLORS = {0: "#2878B5", 1: "#E87500", 2: "#279B70"}

def plot_binary_boundary(X, y, model, feature_names, class_names, title, path):
    X = np.asarray(X)
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, name in enumerate(class_names):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], label=name, alpha=.8,
                   edgecolor="black", linewidth=.4, color=COLORS[label], s=45)
    x_min, x_max = X[:, 0].min() - .3, X[:, 0].max() + .3
    y_min, y_max = X[:, 1].min() - .3, X[:, 1].max() + .3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250),
                         np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, scores >= 0, levels=[-.5, .5, 1.5],
                alpha=.12, colors=[COLORS[0], COLORS[1]])
    if abs(model.weights_[1]) > 1e-12:
        boundary = -(model.weights_[0] * np.linspace(x_min, x_max, 250)
                     + model.bias_) / model.weights_[1]
        ax.plot(np.linspace(x_min, x_max, 250), boundary, "k--",
                linewidth=2, label="Decision boundary")
    else:
        ax.axvline(-model.bias_ / model.weights_[0], color="black",
                   linestyle="--", label="Decision boundary")
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
           xlabel=feature_names[0], ylabel=feature_names[1], title=title)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()      # Opens interactive window
    plt.close(fig)  # Clears memory after you close the window


def plot_learning_curve(model, title, path):
    epochs = np.arange(1, model.n_epochs_ + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, model.mistakes_history_, marker="o")
    axes[0].set(title="Mistakes per epoch", xlabel="Epoch", ylabel="Misclassified samples")
    axes[1].plot(epochs, model.accuracy_history_, marker="o")
    axes[1].set(title="Training accuracy per epoch", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)


def plot_multiclass_regions(X, y, models, class_names, title, path):
    X = np.asarray(X)
    y = np.asarray(y)
    x_min, x_max = X[:, 0].min() - .3, X[:, 0].max() + .3
    y_min, y_max = X[:, 1].min() - .3, X[:, 1].max() + .3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = np.column_stack([m.decision_function(grid) for m in models])
    regions = np.argmax(scores, axis=1).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, regions, levels=np.arange(-.5, len(class_names)+.5, 1),
                alpha=.16, cmap="viridis")
    for idx, name in enumerate(class_names):
        mask = y == idx
        ax.scatter(X[mask, 0], X[mask, 1], label=name, s=42,
                   edgecolor="black", linewidth=.4)
    ax.set(xlabel="PetalLengthCm", ylabel="PetalWidthCm", title=title)
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)


def plot_experiment_comparison(results, path):
    names = [r["experiment"] for r in results]
    train = [r["train_accuracy"] for r in results]
    test = [r["test_accuracy"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = .36
    ax.bar(x - width/2, train, width, label="Train accuracy")
    ax.bar(x + width/2, test, width, label="Test accuracy")
    ax.set_xticks(x, names, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Perceptron experiment comparison")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.show()
    plt.close(fig)