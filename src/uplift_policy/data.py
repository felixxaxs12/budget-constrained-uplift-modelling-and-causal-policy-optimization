"""Data preparation for the official Criteo uplift benchmark."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import yaml


CONTINUOUS_FEATURES = ("f0", "f2", "f7", "f10")
CATEGORICAL_FEATURES = ("f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11")
BINARY_COLUMNS = ("treatment", "conversion", "visit", "exposure")
SOURCE_COLUMNS = (
    "f0",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    *BINARY_COLUMNS,
)
SOURCE_TYPES = {**{name: "DOUBLE" for name in SOURCE_COLUMNS[:12]}, **{name: "UTINYINT" for name in BINARY_COLUMNS}}
SPLITS = ("train", "validation", "test")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the analysis configuration and validate its required contract."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    required = {
        "seed": (),
        "paths": ("raw_data", "processed_data", "model_dir"),
        "features": ("continuous", "categorical", "treatment", "outcomes", "exposure"),
        "split": ("train_buckets", "validation_buckets", "test_buckets", "bucket_count", "hash"),
        "duckdb": ("threads", "memory_limit", "row_group_size"),
    }
    missing: list[str] = []
    for section, keys in required.items():
        if section not in config:
            missing.append(section)
            continue
        for key in keys:
            if key not in config[section]:
                missing.append(f"{section}.{key}")
    if missing:
        raise KeyError(f"Missing required configuration keys: {', '.join(missing)}")

    features = config["features"]
    if tuple(features["continuous"]) != CONTINUOUS_FEATURES:
        raise ValueError("Continuous feature list does not match the source specification")
    if tuple(features["categorical"]) != CATEGORICAL_FEATURES:
        raise ValueError("Categorical feature list does not match the source specification")
    if (features["treatment"], *features["outcomes"], features["exposure"]) != BINARY_COLUMNS:
        raise ValueError("Treatment, outcome, or exposure columns do not match the source specification")

    split = config["split"]
    buckets = [*split["train_buckets"], *split["validation_buckets"], *split["test_buckets"]]
    if split["hash"] != "duckdb_hash" or sorted(buckets) != list(range(split["bucket_count"])):
        raise ValueError("Split buckets must partition the configured DuckDB hash buckets")
    proportions = tuple(
        len(split[name]) / split["bucket_count"]
        for name in ("train_buckets", "validation_buckets", "test_buckets")
    )
    if proportions != (0.6, 0.2, 0.2):
        raise ValueError("The frozen split must be 60/20/20")
    return config


def _connection(config: Mapping[str, Any]) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = ?", [int(config["duckdb"]["threads"])])
    connection.execute("SET memory_limit = ?", [str(config["duckdb"]["memory_limit"])])
    return connection


def _source_scan_sql() -> str:
    schema = ", ".join(f"'{name}': '{dtype}'" for name, dtype in SOURCE_TYPES.items())
    projected = ", ".join(SOURCE_COLUMNS)
    return (
        f"SELECT CAST(ordinality - 1 AS BIGINT) AS row_id, {projected} "
        f"FROM read_csv(?, header = true, columns = {{{schema}}}) WITH ORDINALITY"
    )


def validate_source(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete source without estimating outcome rates or effects."""
    raw_path = Path(config["paths"]["raw_data"])
    with gzip.open(raw_path, mode="rt", encoding="utf-8", newline="") as stream:
        header = tuple(next(csv.reader(stream)))
    if header != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected source header: {header}")

    finite_expression = " AND ".join(f"{name} IS NOT NULL AND isfinite({name})" for name in SOURCE_COLUMNS[:12])
    binary_expression = " AND ".join(
        f"({name} IS NOT NULL AND {name} IN (0, 1))" for name in BINARY_COLUMNS
    )
    query = f"""
        WITH source AS ({_source_scan_sql()})
        SELECT
            count(*)::BIGINT AS row_count,
            bool_and({finite_expression}) AS features_finite,
            bool_and({binary_expression}) AS binary_domains_valid,
            bool_and(treatment = 1 OR exposure = 0) AS control_has_no_exposure,
            bool_and(conversion = 0 OR visit = 1) AS conversion_implies_visit
        FROM source
    """
    connection = _connection(config)
    try:
        row = connection.execute(query, [str(raw_path)]).fetchone()
    finally:
        connection.close()

    result = {
        "header_matches": True,
        "row_count": int(row[0]),
        "features_finite": bool(row[1]),
        "binary_domains_valid": bool(row[2]),
        "control_has_no_exposure": bool(row[3]),
        "conversion_implies_visit": bool(row[4]),
    }
    failed = [name for name, value in result.items() if name != "row_count" and not value]
    if failed:
        raise ValueError(f"Source validation failed: {', '.join(failed)}")
    return result


def _bucket_list(values: Sequence[int]) -> str:
    return ", ".join(str(int(value)) for value in values)


def _partition_query(config: Mapping[str, Any]) -> str:
    split = config["split"]
    source_columns = ", ".join(SOURCE_COLUMNS)
    return f"""
        WITH source AS ({_source_scan_sql()}),
        bucketed AS (
            SELECT *,
                hash(CAST(row_id AS UBIGINT), CAST({int(config['seed'])} AS UBIGINT))
                    % {int(split['bucket_count'])} AS split_bucket
            FROM source
        )
        SELECT row_id, {source_columns},
            CASE
                WHEN split_bucket IN ({_bucket_list(split['train_buckets'])}) THEN 'train'
                WHEN split_bucket IN ({_bucket_list(split['validation_buckets'])}) THEN 'validation'
                WHEN split_bucket IN ({_bucket_list(split['test_buckets'])}) THEN 'test'
            END AS split
        FROM bucketed
    """


def prepare_data(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, split, and write the official data as Hive-partitioned Parquet."""
    validation = validate_source(config)
    raw_path = Path(config["paths"]["raw_data"])
    processed_path = Path(config["paths"]["processed_data"])
    if processed_path.exists():
        raise FileExistsError(f"Processed dataset already exists: {processed_path}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_output = str(processed_path).replace("'", "''")
    copy_sql = f"""
        COPY ({_partition_query(config)}) TO '{escaped_output}' (
            FORMAT PARQUET,
            PARTITION_BY (split),
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE {int(config['duckdb']['row_group_size'])}
        )
    """
    connection = _connection(config)
    try:
        connection.execute(copy_sql, [str(raw_path)])
    finally:
        connection.close()

    train = load_splits(config, ["train"], columns=list(CATEGORICAL_FEATURES))
    category_path = Path(config["paths"]["model_dir"]) / "category_map.json"
    mappings = fit_category_maps(train, CATEGORICAL_FEATURES, output_path=category_path)

    glob = str(processed_path / "split=*" / "*.parquet")
    connection = _connection(config)
    try:
        counts = dict(
            connection.execute(
                "SELECT split, count(*)::BIGINT FROM read_parquet(?, hive_partitioning = true) GROUP BY split",
                [glob],
            ).fetchall()
        )
    finally:
        connection.close()

    if sum(int(count) for count in counts.values()) != validation["row_count"]:
        raise RuntimeError("Partition row counts do not match the validated source")

    manifest = {
        "source": str(raw_path),
        "processed": str(processed_path),
        "format": "hive-partitioned parquet",
        "compression": "zstd",
        "row_id": "zero-based source order",
        "split_hash": "hash(UBIGINT row_id, UBIGINT seed) modulo bucket_count",
        "seed": int(config["seed"]),
        "rows": {name: int(counts[name]) for name in SPLITS},
        "category_map": str(category_path),
        "category_cardinality": {name: len(levels) for name, levels in mappings.items()},
        "source_validation": validation,
    }
    manifest_path = processed_path / "_prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_splits(
    config: Mapping[str, Any],
    splits: Sequence[str],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read selected Hive partitions through DuckDB with a stable row order."""
    invalid_splits = set(splits) - set(SPLITS)
    if invalid_splits or not splits:
        raise ValueError(f"Invalid split selection: {sorted(invalid_splits)}")

    available = {"row_id", *SOURCE_COLUMNS, "split"}
    selected = list(columns) if columns is not None else ["row_id", *SOURCE_COLUMNS, "split"]
    invalid_columns = set(selected) - available
    if invalid_columns:
        raise ValueError(f"Unknown columns: {sorted(invalid_columns)}")

    projection = ", ".join(f'"{name}"' for name in selected)
    placeholders = ", ".join("?" for _ in splits)
    glob = str(Path(config["paths"]["processed_data"]) / "split=*" / "*.parquet")
    query = f"""
        SELECT {projection}
        FROM read_parquet(?, hive_partitioning = true)
        WHERE split IN ({placeholders})
        ORDER BY row_id
    """
    connection = _connection(config)
    try:
        frame = connection.execute(query, [glob, *splits]).fetch_df()
    finally:
        connection.close()

    if "row_id" in frame:
        frame["row_id"] = frame["row_id"].astype("int64", copy=False)
    for name in BINARY_COLUMNS:
        if name in frame:
            frame[name] = frame[name].astype("uint8", copy=False)
    if "split" in frame:
        frame["split"] = pd.Categorical(frame["split"], categories=SPLITS)
    return frame


def fit_category_maps(
    train: pd.DataFrame,
    categorical: Sequence[str],
    output_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """Fit contiguous category codes from training covariates only."""
    if "split" in train and not train["split"].astype("string").eq("train").all():
        raise ValueError("Category levels must be fitted from train rows only")

    mappings: dict[str, list[float]] = {}
    for feature in categorical:
        values = train[feature].to_numpy(dtype=np.float64, copy=False)
        mappings[feature] = np.sort(np.unique(values[np.isfinite(values)])).tolist()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": mappings}, indent=2) + "\n", encoding="utf-8")
    return mappings


def apply_category_maps(
    frame: pd.DataFrame,
    mappings: Mapping[str, Sequence[float]],
    categorical: Sequence[str],
) -> pd.DataFrame:
    """Apply train-fitted maps, representing unseen levels as missing."""
    encoded = frame.copy()
    for feature in categorical:
        lookup = {level: code for code, level in enumerate(mappings[feature])}
        encoded[feature] = encoded[feature].map(lookup).astype("Int32")
    return encoded
