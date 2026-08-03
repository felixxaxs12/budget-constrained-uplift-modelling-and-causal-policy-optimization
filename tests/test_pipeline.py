"""Orchestration tests use explicit fixtures and never produce research results."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
        "format": "hive-partitioned parquet",
        "compression": "zstd",
        "row_id": "zero-based source order",
        "split_hash": "hash(UBIGINT row_id, UBIGINT seed) modulo bucket_count",
        "seed": config["seed"],
        "rows": {"train": 12, "validation": 4, "test": 4},
        "category_map": str(model_dir / "category_map.json"),
        "source_validation": {
            "row_count": 20,
            "header_matches": True,
            "features_finite": True,
            "binary_domains_valid": True,
            "control_has_no_exposure": True,
            "conversion_implies_visit": True,
        },
    }
    (processed / "_prepare_manifest.json").write_text(json.dumps(manifest))


def test_existing_preparation_requires_matching_fingerprints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_preparation_fixture(config)

    with pytest.raises(ValueError, match="does not match"):
        pipeline._matching_preparation(config)

    processed = Path(config["paths"]["processed_data"])
    manifest = json.loads((processed / "_prepare_manifest.json").read_text())
    fingerprinted = pipeline._write_preparation_fingerprints(manifest, config)
    matched = pipeline._matching_preparation(config)
    assert fingerprinted["fingerprints"] == matched["fingerprints"]
    assert set(matched["fingerprints"]) == {
        "preparation_config_sha256",
        "raw_data_sha256",
        "category_map_sha256",
    }

    Path(config["paths"]["raw_data"]).write_bytes(b"changed-source-fixture")
    with pytest.raises(ValueError, match="does not match"):
        pipeline._matching_preparation(config)


def test_freeze_verification_rejects_dirty_training_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_preparation_fixture(config)
    models = Path(config["paths"]["model_dir"])
    tracked = Path(config["paths"]["results_dir"]) / "manifests" / "model_freeze.json"
    tracked.parent.mkdir(parents=True)
    (models / "freeze_manifest.json").write_text("fixture")
    tracked.write_text("fixture")
    manifest = {
        "format_version": 1,
        "source": {"git_dirty": True},
        "hashes": {},
    }
    monkeypatch.setattr(
        pipeline.model_io,
        "load_model_bundle",
        lambda _: SimpleNamespace(manifest=manifest),
    )
    with pytest.raises(ValueError, match="dirty source tree"):
        pipeline._verify_freeze(config)


def test_train_requires_clean_tree_and_copies_exact_freeze_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_preparation_fixture(config)
    model_dir = Path(config["paths"]["model_dir"])
    opaque_development_fixture = object()
    monkeypatch.setattr(pipeline.model_io, "_git_state", lambda: ("commit", False))
    monkeypatch.setattr(pipeline, "_matching_preparation", lambda _: {})
    monkeypatch.setattr(
        pipeline, "load_splits", lambda *_args, **_kwargs: opaque_development_fixture
    )
    monkeypatch.setattr(pipeline, "apply_category_maps", lambda frame, *_args: frame)

    expected_bytes = b'{"source":{"git_dirty":false}}\n'

    def fake_fit(*_args, **_kwargs) -> dict:
        (model_dir / "freeze_manifest.json").write_bytes(expected_bytes)
        return {"source": {"git_dirty": False}}

    monkeypatch.setattr(pipeline.model_io, "fit_model_bundle", fake_fit)
    pipeline.run_train(config)
    tracked = Path(config["paths"]["results_dir"]) / "manifests" / "model_freeze.json"
    assert tracked.read_bytes() == expected_bytes

    monkeypatch.setattr(pipeline.model_io, "_git_state", lambda: ("commit", True))
    monkeypatch.setattr(
        pipeline,
        "load_splits",
        lambda *_args, **_kwargs: pytest.fail("development data must remain unread"),
    )
    with pytest.raises(RuntimeError, match="clean source tree"):
        pipeline.run_train(config)


class _FixtureBundle:
    def __init__(self) -> None:
        self.predictions_created = False
        self.manifest = {
            "features": {"all": [*CONTINUOUS, *CATEGORICAL]},
            "hashes": {
                "raw_data_sha256": "fixture",
                "config_canonical_sha256": "fixture",
                "category_map_sha256": "fixture",
            },
        }

    def predict_policy_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.predictions_created = True
        response = frame["f0"].rank(method="first", pct=True).to_numpy()
        t_score = frame["f2"].rank(method="first", pct=True).to_numpy() - 0.5
        dr_score = frame["f7"].rank(method="first", pct=True).to_numpy()
        return pd.DataFrame(
            {"response": response, "t_learner": t_score, "dr_learner": dr_score}
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
        frame = connection.execute(
            f"""
            SELECT {projection}
            FROM read_parquet(?, hive_partitioning = true)
            WHERE split = ?
            QUALIFY row_number() OVER (PARTITION BY treatment ORDER BY row_id)
                <= ?
            ORDER BY row_id
            """,
            [str(parquet), split, rows_per_arm],
        ).fetch_df()
    finally:
        connection.close()
    return frame


def test_evaluate_keeps_gate_order_and_labels_estimands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_preparation_fixture(config)
    processed = Path(config["paths"]["processed_data"])
    models = Path(config["paths"]["model_dir"])
    (models / "freeze_manifest.json").write_text("{}")
    tracked = Path(config["paths"]["results_dir"]) / "manifests" / "model_freeze.json"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("{}")
    bundle = _FixtureBundle()
    calls: list[str] = []
    feature_columns = ["row_id", *CONTINUOUS, *CATEGORICAL, "split"]
    outcome_columns = ["row_id", "treatment", "conversion", "visit", "split"]
    official_test_features = _official_rows("test", feature_columns)
    official_test_outcomes = _official_rows("test", outcome_columns)
    official_complete_outcomes = pd.concat(
        [
            _official_rows(
                split, ["row_id", "treatment", "conversion", "visit"]
            )
            for split in ("train", "validation", "test")
        ],
        ignore_index=True,
    )

    monkeypatch.setattr(pipeline.model_io, "_git_state", lambda: ("clean-commit", False))
    monkeypatch.setattr(
        pipeline,
        "_matching_preparation",
        lambda _: {"rows": {"train": 12, "validation": 4, "test": 4}},
    )
    monkeypatch.setattr(pipeline, "_verify_freeze", lambda _: bundle)

    def fake_load(_config: dict, splits: list[str], columns: list[str]) -> pd.DataFrame:
        if calls == []:
            assert splits == ["test"]
            assert "treatment" not in columns and "conversion" not in columns
            calls.append("features")
            return official_test_features.copy()
        if calls == ["features"]:
            assert bundle.predictions_created
            assert splits == ["test"]
            calls.append("test_outcomes")
            return official_test_outcomes.copy()
        assert calls == ["features", "test_outcomes"]
        assert splits == ["train", "validation", "test"]
        calls.append("complete_outcomes")
        return official_complete_outcomes.copy()

    monkeypatch.setattr(pipeline, "load_splits", fake_load)
    result = pipeline.run_evaluate(config)

    assert calls == ["features", "test_outcomes", "complete_outcomes"]
    results = Path(config["paths"]["results_dir"])
    ate = pd.read_csv(results / "tables" / "average_treatment_effects.csv")
    policy = pd.read_csv(results / "tables" / "policy_values.csv")
    sample = pd.read_csv(results / "tables" / "sample_summary.csv")
    assert set(ate["sample_scope"]) == {"complete_source"}
    assert (ate["n_treated"] + ate["n_control"]).eq(
        len(official_complete_outcomes)
    ).all()
    assert set(policy["unit"]) == {"incremental_outcomes_per_test_row"}
    assert set(sample["quantity_type"]) == {"sample_count", "sample_rate"}
    run_manifest = json.loads((results / "run_manifest.json").read_text())
    assert run_manifest["source"] == {
        "git_commit_before_outputs": "clean-commit",
        "git_dirty_before_outputs": False,
    }
    assert run_manifest["integrity"][
        "predictions_created_before_test_outcomes_loaded_for_evaluation"
    ]
    assert len(result["figures"]) == 6
    assert (processed / "_prepare_manifest.json").exists()


def test_evaluate_refuses_dirty_tree_before_accessing_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(pipeline.model_io, "_git_state", lambda: ("commit", True))
    monkeypatch.setattr(
        pipeline,
        "load_splits",
        lambda *_args, **_kwargs: pytest.fail("data must remain unread"),
    )
    with pytest.raises(RuntimeError, match="clean source tree"):
        pipeline.run_evaluate(config)


def test_cli_has_only_integrity_compatible_commands() -> None:
    parser = pipeline._parser()
    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["train"]).command == "train"
    assert parser.parse_args(["evaluate"]).command == "evaluate"
    with pytest.raises(SystemExit):
        parser.parse_args(["all"])
