import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_excel("Concrete_Data.xls")

df.columns = [
    "cement", "slag", "fly_ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg",
    "age", "strength"
]

print("Shape:", df.shape)
print(df.head())

print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

print("Shape after dropping duplicates:", df.shape)
print("\nDescribe:\n", df.describe())

df.hist(bins=30, figsize=(14, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("distributions.png")
plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

plt.figure(figsize=(14, 8))
sns.boxplot(data=df.drop(columns=["strength"]))
plt.xticks(rotation=45)
plt.title("Boxplots — Outlier Check")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.close()

X = df.drop(columns=["strength"])
y = df["strength"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

degrees = [1, 2, 3, 4]
results = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)

    cv_scores = cross_val_score(
        LinearRegression(),
        X_train_scaled,
        y_train,
        cv=5,
        scoring="r2"
    )

    results.append({
        "degree": degree,
        "n_features": X_train_poly.shape[1],
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std()
    })

    print(f"\nDegree {degree}:")
    print(f"Features: {X_train_poly.shape[1]}")
    print(f"Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")
    print(f"CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

results_df = pd.DataFrame(results)

print("\n=== Summary Across Degrees ===")
print(results_df)

plt.figure(figsize=(8, 5))
plt.plot(
    results_df["degree"],
    results_df["train_r2"],
    marker="o",
    label="Train R2"
)
plt.plot(
    results_df["degree"],
    results_df["test_r2"],
    marker="o",
    label="Test R2"
)
plt.plot(
    results_df["degree"],
    results_df["cv_r2_mean"],
    marker="o",
    label="CV R2 (mean)"
)
plt.xlabel("Polynomial Degree")
plt.ylabel("R2 Score")
plt.title("Train vs Test vs CV R2 by Polynomial Degree")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("degree_comparison.png")
plt.close()

best_degree = int(
    results_df.loc[results_df["cv_r2_mean"].idxmax(), "degree"]
)

print(f"\nBest Degree Based on CV R2: {best_degree}")

final_poly = PolynomialFeatures(
    degree=best_degree,
    include_bias=False
)

X_train_poly = final_poly.fit_transform(X_train)
X_test_poly = final_poly.transform(X_test)

final_scaler = StandardScaler()

X_train_scaled = final_scaler.fit_transform(X_train_poly)
X_test_scaled = final_scaler.transform(X_test_poly)

final_model = LinearRegression()
final_model.fit(X_train_scaled, y_train)

final_test_pred = final_model.predict(X_test_scaled)

print(
    f"\nFinal Model Test R2: "
    f"{r2_score(y_test, final_test_pred):.4f}"
)
print(
    f"Final Model Test RMSE: "
    f"{np.sqrt(mean_squared_error(y_test, final_test_pred)):.4f}"
)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title(f"Predicted vs Actual (Degree {best_degree})")
plt.tight_layout()
plt.savefig("predicted_vs_actual.png")
plt.close()

print("\nAll plots saved successfully.")