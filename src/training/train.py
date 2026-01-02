import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

DATA_DIR = "/opt/ml/input/data/train"
MODEL_DIR = "/opt/ml/model"


def preprocess(df: pd.DataFrame):
    df = df.copy()
    df["has-capital-gain"] = (df["capital-gain"] > 0).astype(int)
    df["high-hours"] = (df["hours-per-week"] > 45).astype(int)
    df["high-education"] = (df["educational-num"] >= 12).astype(int)

    df["edu-x-hours"] = df["educational-num"] * df["hours-per-week"]
    df["age-x-hours"] = df["age"] * df["hours-per-week"]
    df["log-capital-gain"] = np.log1p(df["capital-gain"])
    df["log-capital-loss"] = np.log1p(df["capital-loss"])
    df["gender-num"] = (df["gender"] == "Male").astype(int)

    df["age-bucket"] = pd.cut(
        df["age"], bins=[0, 25, 40, 60, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    features = [
        "age",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "has-capital-gain",
        "high-hours",
        "high-education",
        "edu-x-hours",
        "age-x-hours",
        "log-capital-gain",
    ]

    df["label"] = (df["income"].str.contains(">50")).astype(int)

    X = df[features]
    y = df["label"]

    return X, y


def main() -> None:
    data_path = os.path.join(DATA_DIR, "adult_income.parquet")
    df = pd.read_parquet(data_path)

    X, y = preprocess(df)

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, random_state=42, learning_rate=0.05
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    recall = recall_score(y_val, preds)

    print(f"validation_recall={recall}")

    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))


if __name__ == "__main__":
    main()
