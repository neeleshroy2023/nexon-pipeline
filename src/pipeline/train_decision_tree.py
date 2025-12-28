import os
from io import BytesIO

import boto3
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
    features = [
        "age",
        "educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
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

    model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=50, random_state=42)

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
