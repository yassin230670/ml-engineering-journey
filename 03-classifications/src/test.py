import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Define the function we looked at earlier
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

# 2. Create mock data
# y_true: What actually happened (3 normal emails, 5 spam emails)
y_true = [0, 1, 1, 0, 1, 0, 1, 1]

# y_pred: What our AI model predicted for each of those 8 emails
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]

# Human-readable names for our two classes
class_names = ["Not Spam", "Spam"]

# 3. Run the function
results = binary_metrics(y_true, y_pred, class_names)

# 4. Print out the results to see what we got
print(f"Accuracy: {results['accuracy'] * 100:.1f}%\n")
print(f"F1 Macro: {results['f1_macro']:.3f}\n")
print("Confusion Matrix:")
print(results['confusion_matrix'])
print("\nDetailed Classification Report:")
print(results['report'])