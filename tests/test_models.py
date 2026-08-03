from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from uplift_policy.data import apply_category_maps, load_config
from uplift_policy.models import _fold_ids, fit_model_bundle, load_model_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS = ["f0", "f2", "f7", "f10"]
CATEGORICAL = ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]
FIXTURE_ROWS_PER_SPLIT = {0: 6_000, 1: 34_000}


@pytest.fixture(scope="session")
def official_development_fixture() -> tuple[dict, pd.DataFrame, Path]:
    raw_path = PROJECT_ROOT / "data/raw/criteo-uplift-v2.1.csv.gz"
    processed_path = PROJECT_ROOT / "data/processed/criteo"
    category_path = PROJECT_ROOT / "artifacts/models/category_map.json"
    partition_files = {
        split: sorted((processed_path / f"split={split}").glob("*.parquet"))
        for split in ("train", "validation")
    }
    if (
        not raw_path.is_file()
        or not category_path.is_file()
        or any(not files for files in partition_files.values())
    ):
        pytest.skip("official raw data and prepared train/validation partitions required")

    selected_columns = [
        "row_id",
        *CONTINUOUS,
        *CATEGORICAL,
        "treatment",
        "conversion",
        "visit",
        "split",
    ]
    projection = ", ".join(f'"{column}"' for column in selected_columns)
    frames: list[pd.DataFrame] = []
    connection = duckdb.connect()
    try:
        for split in ("train", "validation"):
            parquet_glob = str(processed_path / f"split={split}" / "*.parquet")
            for treatment, limit in FIXTURE_ROWS_PER_SPLIT.items():
                frames.append(
                    connection.execute(
                        f"""
                        SELECT {projection}
                        FROM read_parquet(?, hive_partitioning = true)
                        WHERE treatment = ?
                        LIMIT {limit}
                        """,
                        [parquet_glob, treatment],
                    ).fetch_df()
                )
    finally:
        connection.close()

    frame = pd.concat(frames, ignore_index=True)
    with category_path.open(encoding="utf-8") as stream:
        category_maps = json.load(stream)["features"]
    frame = apply_category_maps(frame, category_maps, CATEGORICAL)

    grouped = frame.groupby(["split", "treatment"], observed=True)
    assert grouped.size().eq([6_000, 34_000, 6_000, 34_000]).all()
    assert grouped["conversion"].nunique().eq(2).all()
    assert grouped["visit"].nunique().eq(2).all()

    config = deepcopy(load_config(PROJECT_ROOT / "configs/analysis.yaml"))
    config["paths"]["raw_data"] = str(raw_path)
    config["lightgbm"].update(
        {
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_data_in_leaf": 50,
            "max_bin": 31,
            "max_rounds": 20,
            "early_stopping_rounds": 4,
            "num_threads": 1,
        }
    )
    return config, frame, category_path


def test_fold_assignment_is_seeded_and_deterministic() -> None:
    rows = np.arange(300)
    first = _fold_ids(rows, 3, 20260804)
    second = _fold_ids(rows, 3, 20260804)
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1, 2}
    assert not np.array_equal(first, _fold_ids(rows, 3, 20260805))


def test_fit_save_load_and_predict_bundle(
    tmp_path: Path,
    official_development_fixture: tuple[dict, pd.DataFrame, Path],
) -> None:
    config, frame, category_path = official_development_fixture
    output_dir = tmp_path / "models"

    manifest = fit_model_bundle(config, frame, category_path, output_dir)

    assert set(manifest["models"]) == {
        "conversion_m0",
        "conversion_m1",
        "visit_m0",
        "visit_m1",
        "dr_learner",
    }
    assert set(manifest["selected_rounds"]) == set(manifest["models"])
    assert manifest["folds"] == 3
    assert manifest["propensity"] == 0.85
    assert manifest["hashes"]["raw_data_sha256"]
    assert manifest["source"]["git_commit"]

    loaded = load_model_bundle(output_dir)
    prediction_frame = frame.iloc[:25]
    scores = loaded.predict_policy_scores(prediction_frame)
    assert list(scores) == ["response", "t_learner", "dr_learner"]
    assert scores.shape == (25, 3)
    assert np.isfinite(scores.to_numpy()).all()
    visit_m0, visit_m1 = loaded.predict_nuisance(prediction_frame, "visit")
    assert visit_m0.shape == visit_m1.shape == (25,)
    assert ((visit_m0 >= 0) & (visit_m0 <= 1)).all()
    assert ((visit_m1 >= 0) & (visit_m1 <= 1)).all()


def test_fit_rejects_test_rows(
    tmp_path: Path,
    official_development_fixture: tuple[dict, pd.DataFrame, Path],
) -> None:
    config, frame, category_path = official_development_fixture
    forbidden = frame.copy()
    forbidden.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="only train and validation"):
        fit_model_bundle(config, forbidden, category_path, tmp_path / "models")
