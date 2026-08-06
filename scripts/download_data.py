"""Download and verify the official Criteo Uplift v2.1 dataset."""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from pathlib import Path


SOURCE_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
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


def download() -> tuple[int, str, str]:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = RAW_PATH.with_suffix(RAW_PATH.suffix + ".part")
    digest = hashlib.sha256()
    byte_size = 0
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
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
    return byte_size, observed_sha256, resolved_url


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
    byte_size, sha256, resolved_url = download()
    columns, data_rows = inspect_gzip()
    print(f"Saved {RAW_PATH} ({byte_size:,} bytes)")
    print(f"Rows: {data_rows:,}; columns: {len(columns)}")
    print(f"SHA-256: {sha256}")
    print(f"Resolved URL: {resolved_url}")


if __name__ == "__main__":
    main()
