import os
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

load_dotenv()

BUCKET = os.getenv("ML_BUCKET")
KEY = os.getenv("PARQUET_INPUT_KEY")


def load_data() -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    body = obj["Body"].read()

    df = pd.read_parquet(BytesIO(body))
    return df


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
        "gender-num",
        "high-hours",
        "high-education",
        "edu-x-hours",
        "age-x-hours",
        "log-capital-gain",
    ]

    df["label"] = (df["income"].str.contains(">50")).astype(int)

    X = df[features].values
    y = df["label"]

    return X, y


def main() -> None:
    df = load_data()
    X, y = preprocess(df)

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, min_samples_leaf=20, n_jobs=-1
    )

    model.fit(x_train, y_train)
    preds = model.predict(x_val)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel().tolist()

    print("confusion matrix:")
    print(f"tn: {tn}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    print(f"tp: {tp}")

    r = recall_score(y_val, preds, average="weighted")
    print(r)


if __name__ == "__main__":
    main()
