import pandas as pd
import numpy as np


df = pd.read_csv("car_data.csv")

print("Dataset shape:", df.shape)
print(df.head())


#create branden feature
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

# One hot encoding 
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

X = data.drop(columns=[target]).values
y = data[target].values.astype(float)


#log transform
y = np.log1p(y)


#splitting
np.random.seed(42)

indices = np.random.permutation(len(X))

train_size = int(0.8 * len(X))

train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]



#std
mean = X_train[:, :2].mean(axis=0)
std = X_train[:, :2].std(axis=0)


std[std == 0] = 1

X_train[:, :2] = (
    X_train[:, :2] - mean
) / std

X_test[:, :2] = (
    X_test[:, :2] - mean
) / std



# add bias


X_train = np.c_[
    np.ones(X_train.shape[0]),
    X_train
]

X_test = np.c_[
    np.ones(X_test.shape[0]),
    X_test
]



def train_linear_regression(X,y,learning_rate=0.01,epochs=3000):
    m, n = X.shape
    weights = np.zeros(n)

    for epoch in range(epochs):
        y_pred = X @ weights
        error = y_pred - y
        cost = np.mean(error ** 2)
        gradient = (2 / m) * (X.T @ error)
        weights -= learning_rate * gradient
        # Print progress
        if epoch % 500 == 0:

            print(
                f"Epoch {epoch:4d} | "
                f"Cost: {cost:.6f}"
            )
    return weights



weights = train_linear_regression(X_train, y_train, learning_rate=0.01, epochs=3000)

y_pred_log = X_test @ weights


# Convert predictions back to original price
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)



# MAE
mae = np.mean(np.abs(y_actual - y_pred))
# RMSE
rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
# R²
ss_res = np.sum((y_actual - y_pred) ** 2)

ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)

r2 = 1 - (ss_res / ss_total)



#results
print("\n==============================")
print("results")
print("==============================")

print(f"R²   : {r2:.4f}")
print(f"MAE  : ₹{mae:,.2f}")
print(f"RMSE : ₹{rmse:,.2f}")

#actual Vs predicted
results = pd.DataFrame({
    "Actual Price": y_actual,
    "Predicted Price": y_pred
})

print("\nSample Predictions:")
print(results.head(10))