import os
from io import BytesIO

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("ML_BUCKET")
KEY = os.getenv("INGESTION_FILE_KEY")
PARQUET_OUTPUT_PREFIX = os.getenv("PARQUET_OUTPUT_PREFIX")

s3 = boto3.client("s3")


def main() -> None:
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    body = obj["Body"].read()
    df = pd.read_csv(BytesIO(body), sep=",", on_bad_lines="warn", compression="zip")

    table = pa.Table.from_pandas(df)

    buffer = BytesIO()
    pq.write_table(table, buffer)

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{PARQUET_OUTPUT_PREFIX}data.parquet",
        Body=buffer.getvalue(),
    )

    print(f"Parquet written to s3://{BUCKET}/{PARQUET_OUTPUT_PREFIX}")


if __name__ == "__main__":
    main()
