import hashlib
import os
import zipfile
from io import BytesIO

import boto3
import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.schema import AdultIncomeRow

load_dotenv()

DATA_URL = os.getenv("INGESTION_DATA_URL")
BUCKET = os.getenv("ML_BUCKET")
KEY = os.getenv("INGESTION_FILE_KEY")
ZIP_FILE_NAME = os.getenv("ZIP_FILE_NAME")

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def validate_schema(df: pd.DataFrame):
    for record in df.to_dict(orient="records"):
        AdultIncomeRow(**record)


def checksum(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()


def main() -> None:
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    zip_bytes = response.content

    # unzip in memory
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf, zf.open() as f:
        df = pd.read_csv(
            f, names=COLUMNS, skipinitialspace=True, skiprows=[1], header=0
        )
    df.columns = df.columns.str.replace("-", "_", regex=False)
    validate_schema(df)

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=BUCKET,
        Key=KEY,
        Body=zip_bytes,  # upload original zip, not CSV
        Metadata={
            "rows": str(len(df)),
            "checksum": checksum(zip_bytes),
        },
    )

    print(f"Uploaded {len(df)} rows to S3 bucket with key: /{KEY}")


if __name__ == "__main__":
    main()
