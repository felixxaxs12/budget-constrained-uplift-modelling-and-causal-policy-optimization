"""Pipeline tests use fixtures and never produce reported research results."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from uplift_policy import pipeline


CATEGORICAL = ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]
CONTINUOUS = ["f0", "f2", "f7", "f10"]
ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> dict:
    return {
        "seed": 20260803,
        "propensity": 0.85,
        "capacities": [0.25, 0.5, 1.0],
        "paths": {
            "raw_data": str(tmp_path / "raw.csv.gz"),
            "processed_data": str(tmp_path / "processed"),
            "model_dir": str(tmp_path / "models"),
            "results_dir": str(tmp_path / "results"),
        },
        "features": {
            "continuous": CONTINUOUS,
            "categorical": CATEGORICAL,
            "treatment": "treatment",
            "outcomes": ["conversion", "visit"],
            "exposure": "exposure",
        },
        "split": {
            "train_buckets": list(range(6)),
            "validation_buckets": [6, 7],
            "test_buckets": [8, 9],
            "bucket_count": 10,
            "hash": "duckdb_hash",
        },
        "duckdb": {"threads": 1, "memory_limit": "1GB", "row_group_size": 1000},
        "lightgbm": {"num_threads": 1},
        "dr_learner": {"folds": 3, "fold_seed": 20260804},
        "treatment_diagnostic": {"max_rounds": 5, "early_stopping_rounds": 2},
        "bootstrap": {"replicates": 6, "seed": 20260805},
    }


def _write_preparation_fixture(config: dict) -> None:
    raw, processed, model_dir, _ = pipeline._paths(config)
    raw.write_bytes(b"source-fixture")
    processed.mkdir()
    model_dir.mkdir()
    (model_dir / "category_map.json").write_text(
        json.dumps({"features": {name: [0.0] for name in CATEGORICAL}})
    )
    manifest = {
        "source": str(raw),
        "processed": str(processed),
        "rows": {"train": 12, "validation": 4, "test": 4},
    }
    (processed / "_prepare_manifest.json").write_text(json.dumps(manifest))


def test_train_saves_readable_model_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_preparation_fixture(config)
    development_fixture = object()
    monkeypatch.setattr(
        pipeline, "load_splits", lambda *_args, **_kwargs: development_fixture
    )
    monkeypatch.setattr(pipeline, "apply_category_maps", lambda frame, *_args: frame)

    metadata = {"models": {"conversion_m0": "conversion_m0.txt"}}

    def fake_fit(_config: dict, _frame: object, model_dir: Path) -> dict:
        (model_dir / "model_metadata.json").write_text(json.dumps(metadata))
        return metadata

    monkeypatch.setattr(pipeline.model_io, "fit_model_bundle", fake_fit)
    assert pipeline.run_train(config) == metadata
    saved = Path(config["paths"]["results_dir"]) / "model_metadata.json"
    assert json.loads(saved.read_text()) == metadata


class _FixtureBundle:
    def __init__(self) -> None:
        self.predictions_created = False
        self.manifest = {"features": {"all": [*CONTINUOUS, *CATEGORICAL]}}

    def predict_policy_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.predictions_created = True
        return pd.DataFrame(
            {
                "response": frame["f0"].rank(method="first", pct=True).to_numpy(),
                "t_learner": frame["f2"].rank(method="first", pct=True).to_numpy()
                - 0.5,
                "dr_learner": frame["f7"].rank(method="first", pct=True).to_numpy(),
            }
        )

    def predict_nuisance(
        self, frame: pd.DataFrame, outcome: str
    ) -> tuple[np.ndarray, np.ndarray]:
        assert outcome == "visit"
        base = 0.1 + 0.2 * frame["f10"].rank(method="first", pct=True).to_numpy()
        return base, base + 0.05


def _official_rows(split: str, columns: list[str], rows_per_arm: int = 30) -> pd.DataFrame:
    parquet = ROOT / "data" / "processed" / "criteo" / "split=*" / "*.parquet"
    if not list((ROOT / "data" / "processed" / "criteo").glob("split=*/*.parquet")):
        pytest.skip("Official prepared Criteo data is not present locally")
    projection = ", ".join(f'"{name}"' for name in columns)
    connection = duckdb.connect()
    try:
        return connection.execute(
            f"""
            SELECT {projection}
            FROM read_parquet(?, hive_partitioning = true)
            WHERE split = ?
            QUALIFY row_number() OVER (PARTITION BY treatment ORDER BY row_id) <= ?
            ORDER BY row_id
            """,
            [str(parquet), split, rows_per_arm],
        ).fetch_df()
    finally:
        connection.close()


def test_evaluate_loads_test_outcomes_after_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_preparation_fixture(config)
    bundle = _FixtureBundle()
    calls: list[str] = []
    feature_columns = ["row_id", *CONTINUOUS, *CATEGORICAL, "split"]
    outcome_columns = ["row_id", "treatment", "conversion", "visit", "split"]
    test_features = _official_rows("test", feature_columns)
    test_outcomes = _official_rows("test", outcome_columns)
    complete_outcomes = pd.concat(
        [
            _official_rows(split, ["row_id", "treatment", "conversion", "visit"])
            for split in ("train", "validation", "test")
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(pipeline, "_load_models", lambda _: bundle)

    def fake_load(_config: dict, splits: list[str], columns: list[str]) -> pd.DataFrame:
        if not calls:
            assert "treatment" not in columns and "conversion" not in columns
            calls.append("features")
            return test_features.copy()
        if calls == ["features"]:
            assert bundle.predictions_created
            calls.append("test_outcomes")
            return test_outcomes.copy()
        calls.append("complete_outcomes")
        return complete_outcomes.copy()

    monkeypatch.setattr(pipeline, "load_splits", fake_load)
    result = pipeline.run_evaluate(config)

    assert calls == ["features", "test_outcomes", "complete_outcomes"]
    results = Path(config["paths"]["results_dir"])
    ate = pd.read_csv(results / "tables" / "average_treatment_effects.csv")
    policy = pd.read_csv(results / "tables" / "policy_values.csv")
    assert set(ate["sample_scope"]) == {"complete_source"}
    assert set(policy["sample_scope"]) == {"held_out_test"}
    assert len(result["figures"]) == 3
    assert all(Path(path).suffix == ".png" for path in result["figures"])


def test_cli_commands() -> None:
    parser = pipeline._parser()
    for command in ("prepare", "train", "evaluate"):
        assert parser.parse_args([command]).command == command
