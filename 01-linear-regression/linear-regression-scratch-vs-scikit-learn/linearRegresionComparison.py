import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression



df = pd.read_csv("car_data.csv")

print("Dataset shape:", df.shape)
print(df.head())


#create brand feature
df["brand"] = df["name"].str.split().str[0]
features = [
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "brand"
]
target = "selling_price"
data = df[features + [target]].copy()

#one-hot encoding
data = pd.get_dummies(
    data,
    columns=[
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "brand"
    ],
    dtype=float
)


#split x and y
X = data.drop(columns=[target]).values
y = data[target].values.astype(float)

#log transform target
y = np.log1p(y)



#train test split
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

# 7. Standardization
mean = X_train[:, :2].mean(axis=0)
std = X_train[:, :2].std(axis=0)

std[std == 0] = 1

X_train[:, :2] = (
    X_train[:, :2] - mean
) / std

X_test[:, :2] = (
    X_test[:, :2] - mean
) / std



#add bias
X_train_scratch = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test_scratch = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]



#linear regression from scratch
def train_linear_regression(X, y, learning_rate=0.01, epochs=3000):

    m, n = X.shape

    weights = np.zeros(n)

    for epoch in range(epochs):

        # Prediction
        y_pred = X @ weights

        # Error
        error = y_pred - y

        # Cost
        cost = np.mean(error ** 2)

        # Gradient
        gradient = (2 / m) * (X.T @ error)

        # Update weights
        weights -= learning_rate * gradient

        # Print progress
        if epoch % 500 == 0:

            print(
                f"Epoch {epoch:4d} | "
                f"Cost: {cost:.6f}"
            )

    return weights



print("Training From-Scratch Model")
print("----------------------------")

weights = train_linear_regression(
    X_train_scratch,
    y_train,
    learning_rate=0.01,
    epochs=3000
)



#predictions - from scratch
y_pred_log_scratch = X_test_scratch @ weights
# Convert back from log scale
y_pred_scratch = np.expm1(y_pred_log_scratch)
y_actual = np.expm1(y_test)



#scikit-learn linear regression
print("Training Scikit-Learn Model")
print("--------------------------------")
model = LinearRegression()
model.fit(X_train, y_train)

# predictions
y_pred_log_sklearn = model.predict(X_test)

# convert back from log scale
y_pred_sklearn = np.expm1(y_pred_log_sklearn)


#evaluation function
def evaluate_model(y_actual, y_pred):
    # MAE
    mae = np.mean(np.abs(y_actual - y_pred))
    # RMSE
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    # R²
    ss_res = np.sum((y_actual - y_pred) ** 2)

    ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)

    r2 = 1 - (ss_res / ss_total)

    return r2, mae, rmse



#evaluate both models
r2_scratch, mae_scratch, rmse_scratch = evaluate_model(
    y_actual,
    y_pred_scratch
)

r2_sklearn, mae_sklearn, rmse_sklearn = evaluate_model(
    y_actual,
    y_pred_sklearn
)



#comparison
print("          MODEL COMPARISON")
print("--------------------------------")

print(
    f"{'Metric':<10}"
    f"{'From Scratch':>20}"
    f"{'Scikit-Learn':>20}"
)

print("-" * 50)

print(
    f"{'R²':<10}"
    f"{r2_scratch:>20.4f}"
    f"{r2_sklearn:>20.4f}"
)

print(
    f"{'MAE':<10}"
    f"{mae_scratch:>20,.2f}"
    f"{mae_sklearn:>20,.2f}"
)

print(
    f"{'RMSE':<10}"
    f"{rmse_scratch:>20,.2f}"
    f"{rmse_sklearn:>20,.2f}"
)


#sample predictions
results = pd.DataFrame({

    "Actual Price": y_actual,

    "Scratch Prediction": y_pred_scratch,

    "Sklearn Prediction": y_pred_sklearn

})

print("\nSample Predictions:")
print(results.head(10))