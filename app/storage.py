from functools import lru_cache

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import get_settings


@lru_cache
def _client():
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> None:
    settings = get_settings()
    client = _client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError:
        client.create_bucket(Bucket=settings.s3_bucket)


def put_bytes(key: str, data: bytes, content_type: str) -> None:
    settings = get_settings()
    _client().put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type or "application/octet-stream",
    )


def get_bytes(key: str) -> bytes:
    settings = get_settings()
    response = _client().get_object(Bucket=settings.s3_bucket, Key=key)
    return response["Body"].read()


def open_stream(key: str):
    settings = get_settings()
    response = _client().get_object(Bucket=settings.s3_bucket, Key=key)
    return response["Body"]


def delete_key(key: str) -> None:
    settings = get_settings()
    _client().delete_object(Bucket=settings.s3_bucket, Key=key)


def delete_prefix(prefix: str) -> None:
    settings = get_settings()
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=settings.s3_bucket, Delete={"Objects": objects})
