"""Download and verify the official Criteo Uplift v2.1 dataset."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import ssl
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi


SOURCE_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
DATASET_PAGE = "https://ailab.criteo.com/criteo-uplift-prediction-dataset/"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
EXPECTED_COLUMNS = [
    *(f"f{i}" for i in range(12)),
    "treatment",
    "conversion",
    "visit",
    "exposure",
]
EXPECTED_ROWS = 13_979_592
EXPECTED_BYTE_SIZE = 311_422_618
EXPECTED_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
RAW_PATH = Path("data/raw/criteo-uplift-v2.1.csv.gz")
RECEIPT_PATH = Path("data/raw/download_receipt.json")


def download() -> tuple[int, str, str, str]:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = RAW_PATH.with_suffix(RAW_PATH.suffix + ".part")
    digest = hashlib.sha256()
    byte_size = 0
    retrieved_at = datetime.now(timezone.utc).isoformat()

    tls_context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(SOURCE_URL, timeout=60, context=tls_context) as response:
        resolved_url = response.geturl()
        with partial_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                byte_size += len(chunk)

    observed_sha256 = digest.hexdigest()
    if byte_size != EXPECTED_BYTE_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_BYTE_SIZE:,} bytes, downloaded {byte_size:,} bytes"
        )
    if observed_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Expected SHA-256 {EXPECTED_SHA256}, observed {observed_sha256}"
        )

    os.replace(partial_path, RAW_PATH)
    return byte_size, observed_sha256, retrieved_at, resolved_url


def inspect_gzip() -> tuple[list[str], int]:
    with gzip.open(RAW_PATH, "rt", encoding="utf-8", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split(",")
        data_rows = sum(1 for _ in stream)

    if header != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {header}")
    if data_rows != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {data_rows:,}")
    return header, data_rows


def main() -> None:
    byte_size, sha256, retrieved_at, resolved_url = download()
    columns, data_rows = inspect_gzip()
    manifest = {
        "dataset": "Criteo Uplift Modeling Dataset",
        "release": "CRITEO-UPLIFTv2",
        "artifact_version": "v2.1",
        "publisher": "Criteo AI Lab",
        "file_name": RAW_PATH.name,
        "local_path": RAW_PATH.as_posix(),
        "source_url": SOURCE_URL,
        "resolved_url": resolved_url,
        "dataset_page": DATASET_PAGE,
        "license": "CC BY-NC-SA 4.0",
        "license_url": LICENSE_URL,
        "citations": {
            "criteo_requested": {
                "title": "A Large Scale Benchmark for Uplift Modeling",
                "authors": [
                    "Eustache Diemert",
                    "Artem Betlei",
                    "Christophe Renaudin",
                    "Massih-Reza Amini",
                ],
                "year": 2018,
                "url": "https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018",
            },
            "v2_documentation": {
                "title": "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling",
                "authors": [
                    "Eustache Diemert",
                    "Artem Betlei",
                    "Christophe Renaudin",
                    "Massih-Reza Amini",
                    "Theophane Gregoir",
                    "Thibaud Rahier",
                ],
                "year": 2021,
                "url": "https://arxiv.org/abs/2111.10106",
            },
        },
        "retrieved_at_utc": retrieved_at,
        "source_reported": {
            "compressed_size": "297M",
            "rows": EXPECTED_ROWS,
            "average_visit_rate": 0.046992,
            "average_conversion_rate": 0.00292,
            "treatment_ratio": 0.85,
        },
        "observed": {
            "byte_size": byte_size,
            "expected_byte_size": EXPECTED_BYTE_SIZE,
            "byte_size_matches_expected": byte_size == EXPECTED_BYTE_SIZE,
            "sha256": sha256,
            "expected_sha256": EXPECTED_SHA256,
            "sha256_matches_expected": sha256 == EXPECTED_SHA256,
            "gzip_crc_checked": True,
            "data_rows": data_rows,
            "row_count_matches_source": data_rows == EXPECTED_ROWS,
            "ordered_columns": columns,
        },
        "schema_roles": {
            "feature_columns": {
                "continuous": ["f0", "f2", "f7", "f10"],
                "categorical": ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"],
                "storage": "numeric",
                "role": "pre-treatment covariates",
            },
            "treatment": "source-described randomized assignment",
            "conversion": "binary primary outcome",
            "visit": "binary secondary outcome",
            "exposure": "binary post-assignment exposure; excluded from model features",
        },
    }
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
