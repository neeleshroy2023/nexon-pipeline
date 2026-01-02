import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.image_uris import retrieve

session = sagemaker.Session()
region = session.boto_region_name
role = sagemaker.get_execution_role()

image_uri = retrieve(
    framework="sklearn",
    region=region,
    version="1.2-1",
    py_version="py3",
    instance_type="ml.m5.large",
)

estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    entry_point="train.py",
    source_dir="src/training",
    hyperparameters={},
)

estimator.fit(
    {"train": "s3://mldatalakestack-mldatalakebucketf86ed90f-bpkhb2p5te6a/training/"}
)
