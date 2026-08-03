"""Fit and freeze the pre-specified LightGBM policy models.

The caller supplies features whose categorical columns have already been mapped
by :mod:`uplift_policy.data`.  This module never learns or changes that mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluation import aipw_scores, tie_keys


PREDICTION_BATCH_ROWS = 1_000_000


def _fold_ids(row_ids: Sequence[int] | np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Assign deterministic folds with a chunked SplitMix64 hash."""

    source = np.asarray(row_ids, dtype=np.uint64)
    assigned = np.empty(source.size, dtype=np.uint8)
    for start in range(0, source.size, PREDICTION_BATCH_ROWS):
        stop = min(start + PREDICTION_BATCH_ROWS, source.size)
        assigned[start:stop] = tie_keys(source[start:stop], seed) % np.uint64(folds)
    return assigned


def _feature_names(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    continuous = list(config["features"]["continuous"])
    categorical = list(config["features"]["categorical"])
    return continuous + categorical, categorical


def _lightgbm_params(
    config: Mapping[str, Any], *, objective: str, metric: str, seed: int
) -> dict[str, Any]:
    values = config["lightgbm"]
    return {
        "objective": objective,
        "metric": metric,
        "learning_rate": values["learning_rate"],
        "num_leaves": values["num_leaves"],
        "min_data_in_leaf": values["min_data_in_leaf"],
        "lambda_l2": values["lambda_l2"],
        "feature_fraction": values["feature_fraction"],
        "bagging_fraction": values["bagging_fraction"],
        "max_bin": values["max_bin"],
        "deterministic": values["deterministic"],
        "force_col_wise": values["force_col_wise"],
        "num_threads": values["num_threads"],
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "verbosity": -1,
    }


def _dataset(
    frame: pd.DataFrame,
    row_mask: np.ndarray,
    feature_names: list[str],
    categorical: list[str],
    label: str | np.ndarray,
    *,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    labels = frame.loc[row_mask, label].to_numpy() if isinstance(label, str) else label[row_mask]
    return lgb.Dataset(
        frame.loc[row_mask, feature_names],
        label=labels,
        categorical_feature=categorical,
        reference=reference,
        free_raw_data=True,
    )


def _fit_with_validation(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    label: str | np.ndarray,
    validation_label: str | np.ndarray,
    feature_names: list[str],
    categorical: list[str],
    params: dict[str, Any],
    max_rounds: int,
    stopping_rounds: int,
) -> tuple[lgb.Booster, int, float]:
    train_set = _dataset(
        train, train_mask, feature_names, categorical, label
    )
    validation_set = _dataset(
        validation,
        validation_mask,
        feature_names,
        categorical,
        validation_label,
        reference=train_set,
    )
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(stopping_rounds, verbose=False)],
    )
    selected = int(booster.best_iteration or max_rounds)
    loss = float(booster.best_score["validation"][params["metric"]])
    return booster, selected, loss


def _fit_fixed_rounds(
    frame: pd.DataFrame,
    row_mask: np.ndarray,
    label: str | np.ndarray,
    feature_names: list[str],
    categorical: list[str],
    params: dict[str, Any],
    rounds: int,
) -> lgb.Booster:
    return lgb.train(
        params,
        _dataset(frame, row_mask, feature_names, categorical, label),
        num_boost_round=rounds,
    )


def _predict(
    booster: lgb.Booster,
    frame: pd.DataFrame,
    feature_names: list[str],
    rounds: int | None = None,
) -> np.ndarray:
    predictions = np.empty(len(frame), dtype=np.float64)
    for start in range(0, len(frame), PREDICTION_BATCH_ROWS):
        stop = min(start + PREDICTION_BATCH_ROWS, len(frame))
        predictions[start:stop] = booster.predict(
            frame.iloc[start:stop][feature_names], num_iteration=rounds
        )
    return predictions


def _cross_fitted_conversion_pseudo_outcomes(
    frame: pd.DataFrame,
    feature_names: list[str],
    categorical: list[str],
    nuisance_rounds: Mapping[str, int],
    config: Mapping[str, Any],
) -> np.ndarray:
    folds = int(config["dr_learner"]["folds"])
    fold_seed = int(config["dr_learner"]["fold_seed"])
    propensity = float(config["propensity"])
    fold_id = _fold_ids(frame["row_id"].to_numpy(), folds, fold_seed)
    treatment = frame["treatment"].to_numpy(dtype=np.int8, copy=False)
    outcome = frame["conversion"].to_numpy(dtype=np.int8, copy=False)
    pseudo = np.empty(len(frame), dtype=np.float64)
    params = _lightgbm_params(
        config, objective="binary", metric="binary_logloss", seed=int(config["seed"])
    )

    for fold in range(folds):
        holdout = fold_id == fold
        training = ~holdout
        m0_model = _fit_fixed_rounds(
            frame,
            training & (treatment == 0),
            "conversion",
            feature_names,
            categorical,
            params,
            int(nuisance_rounds["conversion_m0"]),
        )
        m1_model = _fit_fixed_rounds(
            frame,
            training & (treatment == 1),
            "conversion",
            feature_names,
            categorical,
            params,
            int(nuisance_rounds["conversion_m1"]),
        )
        held_out = frame.loc[holdout]
        mu0 = _predict(m0_model, held_out, feature_names)
        mu1 = _predict(m1_model, held_out, feature_names)
        pseudo[holdout] = aipw_scores(
            outcome[holdout], treatment[holdout], mu0, mu1, propensity
        )
    return pseudo


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def _git_state() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


@dataclass
class ModelBundle:
    """Frozen fitted models and their audit manifest."""

    models: dict[str, lgb.Booster]
    manifest: dict[str, Any]

    @property
    def feature_names(self) -> list[str]:
        return list(self.manifest["features"]["all"])

    def predict_nuisance(
        self, frame: pd.DataFrame, outcome: str
    ) -> tuple[np.ndarray, np.ndarray]:
        rounds = self.manifest["selected_rounds"]
        m0_name = f"{outcome}_m0"
        m1_name = f"{outcome}_m1"
        return (
            _predict(self.models[m0_name], frame, self.feature_names, rounds[m0_name]),
            _predict(self.models[m1_name], frame, self.feature_names, rounds[m1_name]),
        )

    def predict_policy_scores(self, frame: pd.DataFrame) -> pd.DataFrame:
        m0, m1 = self.predict_nuisance(frame, "conversion")
        dr = _predict(
            self.models["dr_learner"],
            frame,
            self.feature_names,
            self.manifest["selected_rounds"]["dr_learner"],
        )
        return pd.DataFrame(
            {"response": m1, "t_learner": m1 - m0, "dr_learner": dr},
            index=frame.index,
        )


def fit_model_bundle(
    config: Mapping[str, Any],
    development_df: pd.DataFrame,
    category_map_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Fit all pre-specified models and write a reproducibility freeze manifest."""

    split_values = set(development_df["split"].unique())
    if split_values != {"train", "validation"}:
        raise ValueError("development_df must contain only train and validation rows")
    if float(config["propensity"]) != 0.85:
        raise ValueError("the frozen treatment propensity must equal 0.85")
    if int(config["dr_learner"]["folds"]) != 3:
        raise ValueError("the frozen DR learner must use exactly three folds")

    feature_names, categorical = _feature_names(config)
    train = development_df.loc[development_df["split"] == "train"]
    validation = development_df.loc[development_df["split"] == "validation"]
    train_rows = len(train)
    validation_rows = len(validation)
    train_treatment = train["treatment"].to_numpy(dtype=np.int8, copy=False)
    validation_treatment = validation["treatment"].to_numpy(dtype=np.int8, copy=False)
    seed = int(config["seed"])
    binary_params = _lightgbm_params(
        config, objective="binary", metric="binary_logloss", seed=seed
    )
    max_rounds = int(config["lightgbm"]["max_rounds"])
    stopping_rounds = int(config["lightgbm"]["early_stopping_rounds"])

    selection_models: dict[str, lgb.Booster] = {}
    selected_rounds: dict[str, int] = {}
    validation_losses: dict[str, float] = {}
    for outcome in ("conversion", "visit"):
        for arm in (0, 1):
            name = f"{outcome}_m{arm}"
            model, rounds, loss = _fit_with_validation(
                train,
                validation,
                train_treatment == arm,
                validation_treatment == arm,
                outcome,
                outcome,
                feature_names,
                categorical,
                binary_params,
                max_rounds,
                stopping_rounds,
            )
            if outcome == "conversion":
                selection_models[name] = model
            selected_rounds[name] = rounds
            validation_losses[name] = loss

    train_pseudo = _cross_fitted_conversion_pseudo_outcomes(
        train, feature_names, categorical, selected_rounds, config
    )
    validation_m0 = _predict(
        selection_models["conversion_m0"],
        validation,
        feature_names,
        selected_rounds["conversion_m0"],
    )
    validation_m1 = _predict(
        selection_models["conversion_m1"],
        validation,
        feature_names,
        selected_rounds["conversion_m1"],
    )
    validation_pseudo = aipw_scores(
        validation["conversion"].to_numpy(dtype=np.int8, copy=False),
        validation_treatment,
        validation_m0,
        validation_m1,
        float(config["propensity"]),
    )
    regression_params = _lightgbm_params(
        config, objective="regression", metric="l2", seed=seed
    )
    dr_selection, dr_rounds, dr_loss = _fit_with_validation(
        train,
        validation,
        np.ones(len(train), dtype=bool),
        np.ones(len(validation), dtype=bool),
        train_pseudo,
        validation_pseudo,
        feature_names,
        categorical,
        regression_params,
        max_rounds,
        stopping_rounds,
    )
    del dr_selection
    selected_rounds["dr_learner"] = dr_rounds
    validation_losses["dr_learner"] = dr_loss
    del (
        selection_models,
        train_pseudo,
        validation_m0,
        validation_m1,
        validation_pseudo,
        train_treatment,
        validation_treatment,
        train,
        validation,
    )

    development_pseudo = _cross_fitted_conversion_pseudo_outcomes(
        development_df, feature_names, categorical, selected_rounds, config
    )
    development_treatment = development_df["treatment"].to_numpy(
        dtype=np.int8, copy=False
    )
    final_models: dict[str, lgb.Booster] = {}
    for outcome in ("conversion", "visit"):
        for arm in (0, 1):
            name = f"{outcome}_m{arm}"
            final_models[name] = _fit_fixed_rounds(
                development_df,
                development_treatment == arm,
                outcome,
                feature_names,
                categorical,
                binary_params,
                selected_rounds[name],
            )
    final_models["dr_learner"] = _fit_fixed_rounds(
        development_df,
        np.ones(len(development_df), dtype=bool),
        development_pseudo,
        feature_names,
        categorical,
        regression_params,
        dr_rounds,
    )
    del development_pseudo

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_files: dict[str, dict[str, str]] = {}
    for name, model in final_models.items():
        model_path = destination / f"{name}.txt"
        model.save_model(str(model_path), num_iteration=selected_rounds[name])
        model_files[name] = {
            "path": model_path.name,
            "sha256": _sha256_file(model_path),
        }

    category_path = Path(category_map_path)
    raw_path = Path(config["paths"]["raw_data"])
    commit, dirty = _git_state()
    manifest: dict[str, Any] = {
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": {"all": feature_names, "categorical": categorical},
        "propensity": float(config["propensity"]),
        "seeds": {"model": seed, "fold": int(config["dr_learner"]["fold_seed"])},
        "folds": int(config["dr_learner"]["folds"]),
        "selected_rounds": selected_rounds,
        "validation_losses": validation_losses,
        "policy_scores": {
            "response": "conversion_m1",
            "t_learner": "conversion_m1 - conversion_m0",
            "dr_learner": "dr_learner",
        },
        "models": model_files,
        "hashes": {
            "raw_data_sha256": _sha256_file(raw_path),
            "config_canonical_sha256": _canonical_hash(config),
            "category_map_sha256": _sha256_file(category_path),
        },
        "source": {"git_commit": commit, "git_dirty": dirty},
        "software": {
            "python": platform.python_version(),
            "lightgbm": version("lightgbm"),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
        },
        "development_rows": {
            "train": int(train_rows),
            "validation": int(validation_rows),
            "total": int(len(development_df)),
        },
    }
    manifest_path = destination / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_model_bundle(output_dir: str | Path) -> ModelBundle:
    """Load a frozen bundle, rejecting model files that do not match the manifest."""

    source = Path(output_dir)
    manifest = json.loads((source / "freeze_manifest.json").read_text())
    models: dict[str, lgb.Booster] = {}
    for name, record in manifest["models"].items():
        model_path = source / record["path"]
        if _sha256_file(model_path) != record["sha256"]:
            raise ValueError(f"model hash mismatch: {name}")
        models[name] = lgb.Booster(model_file=str(model_path))
    return ModelBundle(models=models, manifest=manifest)
