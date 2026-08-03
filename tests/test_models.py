from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_policy.evaluation import aipw_scores
from uplift_policy.models import (
    _fold_ids,
    fit_model_bundle,
    load_model_bundle,
)


def _config(raw_path: Path) -> dict:
    return {
        "seed": 20260803,
        "propensity": 0.85,
        "paths": {"raw_data": str(raw_path)},
        "features": {
            "continuous": ["f0", "f2", "f7", "f10"],
            "categorical": ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"],
        },
        "lightgbm": {
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_data_in_leaf": 5,
            "lambda_l2": 1.0,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "max_bin": 31,
            "max_rounds": 20,
            "early_stopping_rounds": 4,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
        },
        "dr_learner": {"folds": 3, "fold_seed": 20260804},
    }


def _development_frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    treatment = (np.arange(rows) % 7 != 0).astype(np.int8)
    continuous = rng.normal(size=(rows, 4)).astype(np.float32)
    probability = 1.0 / (
        1.0 + np.exp(-(-1.2 + 0.7 * continuous[:, 0] + 0.4 * treatment))
    )
    conversion = (rng.random(rows) < probability).astype(np.int8)
    visit = np.maximum(conversion, (rng.random(rows) < 0.35).astype(np.int8))
    frame = pd.DataFrame(
        {
            "row_id": np.arange(rows, dtype=np.int64),
            "split": np.where(np.arange(rows) < 450, "train", "validation"),
            "treatment": treatment,
            "conversion": conversion,
            "visit": visit,
        }
    )
    for index, name in enumerate(["f0", "f2", "f7", "f10"]):
        frame[name] = continuous[:, index]
    for index, name in enumerate(["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]):
        frame[name] = pd.array((np.arange(rows) + index) % 4, dtype="Int32")
    return frame


def test_aipw_pseudo_outcome_matches_formula() -> None:
    treatment = np.array([1, 0, 1, 0])
    outcome = np.array([1, 1, 0, 0])
    m0 = np.array([0.1, 0.2, 0.3, 0.4])
    m1 = np.array([0.5, 0.6, 0.7, 0.8])
    propensity = 0.8
    expected = m1 - m0 + treatment * (outcome - m1) / propensity - (
        1 - treatment
    ) * (outcome - m0) / (1 - propensity)
    np.testing.assert_allclose(
        aipw_scores(outcome, treatment, m0, m1, propensity), expected
    )


def test_fold_assignment_is_seeded_and_deterministic() -> None:
    rows = np.arange(300)
    first = _fold_ids(rows, 3, 20260804)
    second = _fold_ids(rows, 3, 20260804)
    np.testing.assert_array_equal(first, second)
    assert set(first) == {0, 1, 2}
    assert not np.array_equal(first, _fold_ids(rows, 3, 20260805))


def test_fit_save_load_and_predict_bundle(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv.gz"
    raw_path.write_bytes(b"official-source-placeholder-for-unit-test")
    category_path = tmp_path / "category_maps.json"
    category_path.write_text(
        json.dumps(
            {
                "features": {
                    name: [0.0, 1.0, 2.0, 3.0]
                    for name in ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]
                }
            }
        )
    )
    config = _config(raw_path)
    frame = _development_frame()
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


def test_fit_rejects_test_rows(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv.gz"
    raw_path.write_bytes(b"fixture")
    category_path = tmp_path / "category_maps.json"
    category_path.write_text('{"features": {}}')
    frame = _development_frame()
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="only train and validation"):
        fit_model_bundle(_config(raw_path), frame, category_path, tmp_path / "models")
