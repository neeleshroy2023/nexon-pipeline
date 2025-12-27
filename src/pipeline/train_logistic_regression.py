import os
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("ML_BUCKET")
PARQUET_KEY = os.getenv("PARQUET_INPUT_KEY")


def load_data() -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=PARQUET_KEY)
    body = obj["Body"].read()
    df = pd.read_parquet(BytesIO(body))
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["label"] = (df["income"].str.contains(">50")).astype(int)

    features = [
        "age",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    X = df[features].values
    y = df["label"].values

    return X, y


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def train(X, y, lr=0.005, epochs=1000):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for i in range(epochs):
        z: np.float128 = X @ w + b
        p: np.float128 = sigmoid(z)

        dw = (1 / n) * X.T @ (p - y)
        db = (1 / n) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

        if i % 100 == 0:
            loss = -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean()
            print(f"epoch={i}, loss={loss:.4f}")

    return w, b


def train_val_split(X, y, val_ratio=0.2):
    n = len(X)
    idx = int(n * (1 - val_ratio))
    return X[:idx], X[idx:], y[:idx], y[idx:]


def evaluate(X, y, w, b):
    p = sigmoid(X @ w + b)
    preds = (p > 0.5).astype(int)
    accuracy = (preds == y).mean()
    return accuracy


def confusion_matrix(X, y, w, b):
    p = sigmoid(X @ w + b)
    preds = (p > 0.5).astype(int)

    tp = ((preds == 1) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()

    return tp, fp, fn, tn


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std


def main() -> None:
    df = load_data()
    X, y = preprocess(df)
    X = standardize(X)
    x_train, x_val, y_train, y_val = train_val_split(X, y)
    w, b = train(x_train, y_train)

    acc = evaluate(x_val, y_val, w, b)
    print("validation accuracy:", acc)

    tp, fp, fn, tn = confusion_matrix(x_val, y_val, w, b)
    print("TP, FP, FN, TN:", tp, fp, fn, tn)


if __name__ == "__main__":
    main()
