"""Development-only randomization diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _require_development_rows(frame: pd.DataFrame) -> None:
    if "split" in frame and frame["split"].astype("string").eq("test").any():
        raise ValueError("Development diagnostics are restricted to train and validation rows")


def _standardized_difference(difference: float, scale: float) -> float:
    if scale > 0:
        return difference / scale
    if difference == 0:
        return 0.0
    return float(np.copysign(np.inf, difference))


def balance_table(
    development: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
    treatment_col: str = "treatment",
) -> pd.DataFrame:
    """Return one pre-specified balance statistic per covariate."""
    _require_development_rows(development)
    treated = development[treatment_col].eq(1)
    rows: list[dict[str, str | float]] = []

    for feature in continuous:
        treated_values = development.loc[treated, feature]
        control_values = development.loc[~treated, feature]
        difference = float(treated_values.mean() - control_values.mean())
        pooled_sd = float(np.sqrt((treated_values.var(ddof=1) + control_values.var(ddof=1)) / 2.0))
        rows.append(
            {
                "feature": feature,
                "metric_type": "pooled_smd",
                "value": _standardized_difference(difference, pooled_sd),
            }
        )

    for feature in categorical:
        treated_rates = development.loc[treated, feature].value_counts(normalize=True)
        control_rates = development.loc[~treated, feature].value_counts(normalize=True)
        levels = treated_rates.index.union(control_rates.index)
        treated_rates = treated_rates.reindex(levels, fill_value=0.0)
        control_rates = control_rates.reindex(levels, fill_value=0.0)
        difference = treated_rates - control_rates
        pooled_sd = np.sqrt(
            (treated_rates * (1.0 - treated_rates) + control_rates * (1.0 - control_rates)) / 2.0
        )
        values = [
            abs(_standardized_difference(float(delta), float(scale)))
            for delta, scale in zip(difference, pooled_sd, strict=True)
        ]
        rows.append(
            {
                "feature": feature,
                "metric_type": "max_abs_one_vs_rest_smd",
                "value": max(values),
            }
        )

    return pd.DataFrame(rows, columns=["feature", "metric_type", "value"])


def treatment_auc(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical: Sequence[str] = (),
    treatment_col: str = "treatment",
    *,
    params: Mapping[str, object] | None = None,
    max_rounds: int = 300,
    early_stopping_rounds: int = 30,
) -> float:
    """Fit a covariate-only treatment classifier and return validation AUC."""
    _require_development_rows(train)
    _require_development_rows(validation)
    forbidden = {treatment_col, "conversion", "visit", "exposure", "row_id", "split"}
    if forbidden.intersection(feature_columns):
        raise ValueError("Treatment diagnostic features must contain pre-treatment covariates only")

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    model_params: dict[str, object] = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 1000,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "n_estimators": max_rounds,
    }
    if params is not None:
        model_params.update(params)
    model_params["n_estimators"] = max_rounds

    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        train[list(feature_columns)],
        train[treatment_col],
        categorical_feature=list(categorical),
        eval_X=validation[list(feature_columns)],
        eval_y=validation[treatment_col],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    probabilities = model.predict_proba(validation[list(feature_columns)])[:, 1]
    return float(roc_auc_score(validation[treatment_col], probabilities))
