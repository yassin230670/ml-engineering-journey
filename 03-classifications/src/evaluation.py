import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
def binary_metrics(y_true, y_pred, class_names):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=class_names,
            zero_division=0, digits=3
        ),
    }
def multiclass_metrics(y_true, y_pred, class_names):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        )[0]),
        "recall_macro": float(precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        )[1]),
        "f1_macro": float(precision_recall_fscore_support(
            y_true, y_pred, labels=class_names, average="macro", zero_division=0
        )[2]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=class_names),
        "report": classification_report(
            y_true, y_pred, labels=class_names, target_names=class_names,
            zero_division=0, digits=3
        ),
    }
