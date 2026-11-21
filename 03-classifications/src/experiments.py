import numpy as np
from .perceptron import Perceptron
from .data_preprocessing import (FEATURES, SPECIES, binary_data,
                                 split_and_scale, multiclass_split_and_scale)
from .evaluation import binary_metrics, multiclass_metrics
from .visualization import (plot_binary_boundary, plot_learning_curve,
                            plot_confusion, plot_multiclass_regions,
                            plot_experiment_comparison)

PLOT_FEATURES = ["PetalLengthCm", "PetalWidthCm"]


def train_binary_experiment(df, class_0, class_1, name, results_dir, seed=42):
    subset, X, y = binary_data(df, class_0, class_1)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y, random_state=seed)
    model = Perceptron(learning_rate=0.1, max_epochs=100, random_state=seed)
    model.fit(X_train, y_train)
    train_pred, test_pred = model.predict(X_train), model.predict(X_test)
    metrics = binary_metrics(y_test, test_pred, [class_0, class_1])

    # Train a distinct two-feature model for the 2D decision-boundary visualization.
    X2 = subset[PLOT_FEATURES].to_numpy(dtype=float)
    _, X2_test, y2_train, y2_test, scaler2 = split_and_scale(X2, y, random_state=seed)
    # Fit the plotting model on the full subset so its displayed boundary represents all available samples.
    X2_scaled = scaler2.fit_transform(X2)
    plot_model = Perceptron(learning_rate=0.1, max_epochs=100, random_state=seed)
    plot_model.fit(X2_scaled, y)
    plot_binary_boundary(X2_scaled, y, plot_model, PLOT_FEATURES, [class_0, class_1],
                         f"{name}: 2-feature decision boundary (scaled)", 
                         results_dir / f"{name}_decision_boundary.png")
    plot_learning_curve(model, f"{name}: 4-feature model", results_dir / f"{name}_learning_curve.png")

    result = {
        "experiment": name,
        "train_accuracy": float(np.mean(train_pred == y_train)),
        "test_accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "epochs": model.n_epochs_,
        "converged": model.converged_,
        "final_epoch_mistakes": model.mistakes_history_[-1],
        "model": model,
        "metrics": metrics,
        "class_names": [class_0, class_1],
    }
    print_binary_report(result)
    return result


def print_binary_report(result):
    print(f"\n{'='*72}\n{result['experiment']}\n{'='*72}")
    for key in ["train_accuracy", "test_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"{key.replace('_', ' ').title():25s}: {result[key]:.3f}")
    print(f"Epochs completed         : {result['epochs']}")
    print(f"Converged (zero mistakes): {result['converged']}")
    print(f"Final epoch mistakes     : {result['final_epoch_mistakes']}")
    print("\nClassification report:\n", result["metrics"]["report"])


def run_ovr(df, results_dir, seed=42):
    X_train, X_test, y_train_labels, y_test_labels, scaler = multiclass_split_and_scale(df, random_state=seed)
    models = []
    per_model_info = []
    for idx, positive_class in enumerate(SPECIES):
        y_binary = (y_train_labels == positive_class).astype(int)
        model = Perceptron(learning_rate=0.1, max_epochs=150, random_state=seed + idx)
        model.fit(X_train, y_binary)
        models.append(model)
        per_model_info.append({
            "class": positive_class,
            "epochs": model.n_epochs_,
            "converged": model.converged_,
            "final_epoch_mistakes": model.mistakes_history_[-1],
            "mistakes_history": model.mistakes_history_,
        })

    score_matrix_train = np.column_stack([m.decision_function(X_train) for m in models])
    score_matrix_test = np.column_stack([m.decision_function(X_test) for m in models])
    train_pred = np.array(SPECIES)[np.argmax(score_matrix_train, axis=1)]
    test_pred = np.array(SPECIES)[np.argmax(score_matrix_test, axis=1)]
    metrics = multiclass_metrics(y_test_labels, test_pred, SPECIES)

    # Two-feature visualization-only OVR models, trained independently on the petal features.
    X2 = df[PLOT_FEATURES].to_numpy(dtype=float)
    y2 = df["Species"].to_numpy()
    from sklearn.model_selection import train_test_split
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=seed, stratify=y2
    )
    from sklearn.preprocessing import StandardScaler
    scaler2 = StandardScaler().fit(X2_train)
    X2_train = scaler2.transform(X2_train)
    X2_all_scaled = scaler2.transform(X2)
    plot_models = []
    for idx, positive_class in enumerate(SPECIES):
        m = Perceptron(learning_rate=0.1, max_epochs=150, random_state=seed + idx)
        m.fit(X2_train, (y2_train == positive_class).astype(int))
        plot_models.append(m)
    plot_multiclass_regions(
        X2_all_scaled, np.array([SPECIES.index(v) for v in y2]),
        plot_models, SPECIES,
        "One-vs-Rest regions (2-feature visualization model)",
        results_dir / "experiment3_decision_regions.png"
    )
    plot_confusion(metrics["confusion_matrix"], SPECIES, "One-vs-Rest confusion matrix",
                   results_dir / "experiment3_confusion_matrix.png")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    for info in per_model_info:
        ax.plot(range(1, len(info["mistakes_history"]) + 1), info["mistakes_history"],
                marker="o", markersize=3, label=f"{info['class']} vs rest")
    ax.set(title="One-vs-Rest: mistakes per epoch", xlabel="Epoch", ylabel="Mistakes")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "experiment3_learning_curves.png", dpi=160)
    plt.close(fig)

    result = {
        "experiment": "One-vs-Rest (3 classes)",
        "train_accuracy": float(np.mean(train_pred == y_train_labels)),
        "test_accuracy": metrics["accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "epochs": max(info["epochs"] for info in per_model_info),
        "converged": all(info["converged"] for info in per_model_info),
        "final_epoch_mistakes": sum(info["final_epoch_mistakes"] for info in per_model_info),
        "metrics": metrics,
        "models": models,
        "per_model_info": per_model_info,
    }
    print(f"\n{'='*72}\nOne-vs-Rest (3 classes)\n{'='*72}")
    for key in ["train_accuracy", "test_accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"{key.replace('_', ' ').title():25s}: {result[key]:.3f}")
    print("\nPer-class model convergence:")
    for info in per_model_info:
        print(f"  {info['class']}: epochs={info['epochs']}, converged={info['converged']}, "
              f"final mistakes={info['final_epoch_mistakes']}")
    print("\nClassification report:\n", metrics["report"])
    return result


def save_comparison(results, results_dir):
    rows = []
    for r in results:
        rows.append({k: r[k] for k in [
            "experiment", "train_accuracy", "test_accuracy",
            "precision_macro", "recall_macro", "f1_macro", "epochs",
            "converged", "final_epoch_mistakes"
        ]})
    import pandas as pd
    comparison = pd.DataFrame(rows)
    comparison.to_csv(results_dir / "experiment_comparison.csv", index=False)
    comparison.to_markdown(results_dir / "experiment_comparison.md", index=False, floatfmt=".3f")
    plot_experiment_comparison(results, results_dir / "experiment_comparison.png")
    print(f"\n{'='*72}\nCOMPARISON TABLE\n{'='*72}")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
