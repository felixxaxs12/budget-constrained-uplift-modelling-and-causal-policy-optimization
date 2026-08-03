from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from uplift_policy.audit import balance_table, treatment_auc
from uplift_policy.data import (
    CATEGORICAL_FEATURES,
    SOURCE_COLUMNS,
    _source_scan_sql,
    apply_category_maps,
    fit_category_maps,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_contract() -> None:
    config = load_config(ROOT / "configs" / "analysis.yaml")
    assert config["seed"] == 20260803
    assert config["split"]["train_buckets"] == [0, 1, 2, 3, 4, 5]


def test_official_source_has_explicit_schema_and_stable_ordinality() -> None:
    raw_path = ROOT / "data" / "raw" / "criteo-uplift-v2.1.csv.gz"
    if not raw_path.exists():
        pytest.skip("Official Criteo artifact is not present locally")

    connection = duckdb.connect()
    try:
        relation = connection.execute(
            f"SELECT * FROM ({_source_scan_sql()}) LIMIT 3", [str(raw_path)]
        )
        frame = relation.fetch_df()
    finally:
        connection.close()

    assert frame.columns.tolist() == ["row_id", *SOURCE_COLUMNS]
    assert frame["row_id"].tolist() == [0, 1, 2]
    assert all(str(dtype) == "float64" for dtype in frame.iloc[:, 1:13].dtypes)
    assert all(str(dtype) == "uint8" for dtype in frame.iloc[:, 13:].dtypes)


def test_binary_domain_contract_rejects_null() -> None:
    expression = " AND ".join(
        f"({name} IS NOT NULL AND {name} IN (0, 1))"
        for name in ("treatment", "conversion", "visit", "exposure")
    )
    connection = duckdb.connect()
    try:
        valid = connection.execute(
            f"""
            SELECT bool_and({expression})
            FROM (VALUES (0, 0, 0, 0), (NULL, 0, 0, 0))
                AS rows(treatment, conversion, visit, exposure)
            """
        ).fetchone()[0]
    finally:
        connection.close()
    assert valid is False


def test_train_category_maps_are_contiguous_and_unseen_is_missing(tmp_path: Path) -> None:
    train = pd.DataFrame({"f1": [3.0, 1.0, 3.0, 2.0], "split": ["train"] * 4})
    output = tmp_path / "category_map.json"
    mappings = fit_category_maps(train, ["f1"], output)
    encoded = apply_category_maps(pd.DataFrame({"f1": [1.0, 3.0, 9.0]}), mappings, ["f1"])

    assert mappings == {"f1": [1.0, 2.0, 3.0]}
    assert encoded["f1"].tolist() == [0, 2, pd.NA]
    assert str(encoded["f1"].dtype) == "Int32"
    assert output.read_text(encoding="utf-8").startswith('{\n  "features"')


def test_category_map_rejects_non_train_rows() -> None:
    frame = pd.DataFrame({"f1": [1.0, 2.0], "split": ["train", "validation"]})
    with pytest.raises(ValueError, match="train rows only"):
        fit_category_maps(frame, ["f1"])


def test_balance_table_matches_algebraic_smds() -> None:
    development = pd.DataFrame(
        {
            "x": [3.0, 5.0, 1.0, 3.0],
            "g": [1, 1, 1, 0],
            "treatment": [1, 1, 0, 0],
            "split": ["train", "validation", "train", "validation"],
        }
    )
    result = balance_table(development, continuous=["x"], categorical=["g"])

    assert result.loc[result["feature"].eq("x"), "value"].item() == pytest.approx(np.sqrt(2.0))
    assert result.loc[result["feature"].eq("g"), "value"].item() == pytest.approx(
        0.5 / np.sqrt(0.125)
    )


def test_balance_table_rejects_test_rows() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0], "treatment": [0, 1], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="train and validation"):
        balance_table(frame, continuous=["x"], categorical=[])


def test_treatment_classifier_rejects_post_assignment_features() -> None:
    frame = pd.DataFrame({"conversion": [0, 1], "treatment": [0, 1]})
    with pytest.raises(ValueError, match="pre-treatment covariates only"):
        treatment_auc(frame, frame, feature_columns=["conversion"])


def test_treatment_classifier_auc_uses_encoded_development_frames() -> None:
    rng = np.random.default_rng(20260803)
    x_train = rng.normal(size=400)
    x_validation = rng.normal(size=200)
    train = pd.DataFrame({"x": x_train, "treatment": (x_train > 0).astype("uint8")})
    validation = pd.DataFrame({"x": x_validation, "treatment": (x_validation > 0).astype("uint8")})

    auc = treatment_auc(
        train,
        validation,
        feature_columns=["x"],
        params={"min_child_samples": 5, "num_leaves": 7, "n_jobs": 1},
        max_rounds=30,
        early_stopping_rounds=5,
    )
    assert auc > 0.98


def test_all_source_categorical_fields_are_covered() -> None:
    assert set(CATEGORICAL_FEATURES) == {"f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"}
