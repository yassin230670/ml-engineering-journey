from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_iris_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace Iris.csv in the project's data/ folder "
            "or pass its path with --data."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    required = FEATURES + ["Species"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    df = df[required].copy()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Species"] = df["Species"].astype(str).str.strip()
    df["Species"] = df["Species"].str.replace("_", "-", regex=False)
    bad_numeric = df[FEATURES].isna().any(axis=1)
    bad_species = ~df["Species"].isin(SPECIES)
    if bad_numeric.any() or bad_species.any():
        raise ValueError(
            f"Invalid rows found: {int(bad_numeric.sum())} rows with missing/non-numeric "
            f"features; {int(bad_species.sum())} rows with unknown species labels. "
            "Inspect and clean the CSV rather than silently dropping data."
        )
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains non-finite feature values.")
    if df.duplicated().any():
        print(f"Warning: {int(df.duplicated().sum())} duplicate rows found (retained).")
    return df


def binary_data(df, class_0, class_1):
    subset = df[df["Species"].isin([class_0, class_1])].copy()
    X = subset[FEATURES].to_numpy(dtype=float)
    y = (subset["Species"].to_numpy() == class_1).astype(int)
    return subset, X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def multiclass_split_and_scale(df, test_size=0.2, random_state=42):
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["Species"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    #normairizes data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
