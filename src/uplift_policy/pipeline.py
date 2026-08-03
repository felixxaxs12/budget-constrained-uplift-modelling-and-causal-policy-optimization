"""Command-line orchestration for the reproducible uplift-policy analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import models as model_io
from .audit import balance_table, treatment_auc
from .data import (
    apply_category_maps,
    load_config,
    load_splits,
    prepare_data,
)
from .evaluation import aipw_scores, binary_ate, paired_row_bootstrap, qini_curve


def _paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    raw = Path(config["paths"]["raw_data"])
    processed = Path(config["paths"]["processed_data"])
    model_dir = Path(config["paths"]["model_dir"])
    results = Path(config["paths"]["results_dir"])
    return raw, processed, model_dir, results


def _preparation_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only settings that determine prepared data and category codes."""
    return {
        "seed": config["seed"],
        "paths": {
            name: config["paths"][name]
            for name in ("raw_data", "processed_data", "model_dir")
        },
        "features": config["features"],
        "split": config["split"],
        "duckdb": config["duckdb"],
    }


def _category_maps(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mappings = payload.get("features")
    if not isinstance(mappings, dict):
        raise ValueError(f"Invalid category map: {path}")
    return mappings


def _write_preparation_fingerprints(
    manifest: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    raw, processed, model_dir, _ = _paths(config)
    category_path = model_dir / "category_map.json"
    manifest["fingerprints"] = {
        "preparation_config_sha256": model_io._canonical_hash(
            _preparation_spec(config)
        ),
        "raw_data_sha256": model_io._sha256_file(raw),
        "category_map_sha256": model_io._sha256_file(category_path),
    }
    (processed / "_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _matching_preparation(config: Mapping[str, Any]) -> dict[str, Any]:
    raw, processed, model_dir, _ = _paths(config)
    manifest_path = processed / "_prepare_manifest.json"
    category_path = model_dir / "category_map.json"
    if not manifest_path.is_file() or not category_path.is_file():
        raise FileNotFoundError("Prepared data or its category map is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "preparation_config_sha256": model_io._canonical_hash(
            _preparation_spec(config)
        ),
        "raw_data_sha256": model_io._sha256_file(raw),
        "category_map_sha256": model_io._sha256_file(category_path),
    }
    if manifest.get("fingerprints") != expected:
        raise ValueError(
            "Existing prepared data does not match the current source or preparation settings"
        )
    if Path(manifest["source"]).resolve() != raw.resolve():
        raise ValueError("Prepared-data source path does not match the configuration")
    if Path(manifest["processed"]).resolve() != processed.resolve():
        raise ValueError("Prepared-data output path does not match the configuration")
    return manifest


def _save_figure(figure: Any, stem: Path, *, pdf: bool) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    outputs = [png]
    if pdf:
        pdf_path = stem.with_suffix(".pdf")
        figure.savefig(pdf_path, bbox_inches="tight")
        outputs.append(pdf_path)
    return outputs


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_balance(balance: pd.DataFrame, output: Path) -> None:
    plt = _pyplot()

    ordered = balance.assign(abs_value=balance["value"].abs()).sort_values(
        "abs_value"
    )
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    colors = np.where(
        ordered["metric_type"].eq("pooled_smd"), "#3B6FB6", "#E07A3F"
    )
    axis.barh(ordered["feature"], ordered["value"], color=colors)
    axis.axvline(0.1, color="0.35", linewidth=1, linestyle="--")
    axis.axvline(-0.1, color="0.35", linewidth=1, linestyle="--")
    axis.axvline(0.0, color="0.2", linewidth=0.8)
    axis.set_xlabel("Standardized mean difference")
    axis.set_ylabel("")
    axis.set_title("Development-sample covariate balance")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    _save_figure(figure, output, pdf=False)
    plt.close(figure)


def run_prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare the source and write development-only randomization diagnostics."""
    _, processed, model_dir, results = _paths(config)
    manifest_path = processed / "_prepare_manifest.json"
    if manifest_path.exists():
        manifest = _matching_preparation(config)
    else:
        manifest = _write_preparation_fingerprints(prepare_data(config), config)

    feature_names = [
        *config["features"]["continuous"],
        *config["features"]["categorical"],
    ]
    columns = ["row_id", *feature_names, "treatment", "split"]
    train = load_splits(config, ["train"], columns=columns)
    validation = load_splits(config, ["validation"], columns=columns)
    development = pd.concat([train, validation], ignore_index=True)
    balance = balance_table(
        development,
        config["features"]["continuous"],
        config["features"]["categorical"],
    )
    balance.insert(2, "absolute_value", balance["value"].abs())
    balance["sample_scope"] = "train_and_validation"
    balance["unit"] = "standardized_mean_difference"
    del development

    mappings = _category_maps(model_dir / "category_map.json")
    train = apply_category_maps(
        train, mappings, config["features"]["categorical"]
    )
    validation = apply_category_maps(
        validation, mappings, config["features"]["categorical"]
    )
    auc = treatment_auc(
        train,
        validation,
        feature_names,
        config["features"]["categorical"],
        max_rounds=int(config["treatment_diagnostic"]["max_rounds"]),
        early_stopping_rounds=int(
            config["treatment_diagnostic"]["early_stopping_rounds"]
        ),
        params={
            "random_state": int(config["seed"]),
            "n_jobs": int(config["lightgbm"]["num_threads"]),
        },
    )

    audit_rows: list[dict[str, Any]] = []
    for name, value in manifest["source_validation"].items():
        audit_rows.append(
            {
                "metric": name,
                "value": value,
                "quantity_type": "sample_count" if name == "row_count" else "integrity_check",
                "unit": "rows" if name == "row_count" else "boolean",
                "sample_scope": "complete_source",
            }
        )
    for split, count in manifest["rows"].items():
        audit_rows.append(
            {
                "metric": f"{split}_rows",
                "value": count,
                "quantity_type": "sample_count",
                "unit": "rows",
                "sample_scope": split,
            }
        )
    audit_rows.extend(
        [
            {
                "metric": "treatment_classifier_auc",
                "value": auc,
                "quantity_type": "descriptive_randomization_diagnostic",
                "unit": "auc",
                "sample_scope": "train_to_validation",
            },
            {
                "metric": "maximum_absolute_smd",
                "value": float(balance["absolute_value"].max()),
                "quantity_type": "descriptive_randomization_diagnostic",
                "unit": "standardized_mean_difference",
                "sample_scope": "train_and_validation",
            },
        ]
    )

    tables = results / "tables"
    figures = results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(tables / "data_audit.csv", index=False)
    balance.to_csv(tables / "balance.csv", index=False)
    _plot_balance(balance, figures / "balance")
    return {
        "prepared_rows": manifest["rows"],
        "treatment_classifier_auc": auc,
        "maximum_absolute_smd": float(balance["absolute_value"].max()),
    }


def run_train(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fit the frozen policy and nuisance models using development rows only."""
    _, dirty = model_io._git_state()
    if dirty:
        raise RuntimeError("Model fitting requires a clean source tree")
    _matching_preparation(config)
    _, _, model_dir, results = _paths(config)
    feature_names = [
        *config["features"]["continuous"],
        *config["features"]["categorical"],
    ]
    columns = [
        "row_id",
        *feature_names,
        "treatment",
        *config["features"]["outcomes"],
        "split",
    ]
    development = load_splits(
        config, ["train", "validation"], columns=columns
    )
    category_path = model_dir / "category_map.json"
    development = apply_category_maps(
        development,
        _category_maps(category_path),
        config["features"]["categorical"],
    )
    manifest = model_io.fit_model_bundle(
        config, development, category_path, model_dir
    )
    private_manifest = model_dir / "freeze_manifest.json"
    tracked_manifest = results / "manifests" / "model_freeze.json"
    tracked_manifest.parent.mkdir(parents=True, exist_ok=True)
    tracked_manifest.write_bytes(private_manifest.read_bytes())
    return manifest


def _verify_freeze(config: Mapping[str, Any]) -> model_io.ModelBundle:
    """Verify every frozen input and fitted-model hash without reading outcomes."""
    raw, _, model_dir, results = _paths(config)
    category_path = model_dir / "category_map.json"
    private_manifest = model_dir / "freeze_manifest.json"
    tracked_manifest = results / "manifests" / "model_freeze.json"
    if not private_manifest.is_file() or not tracked_manifest.is_file():
        raise FileNotFoundError("Private or tracked model-freeze manifest is missing")
    if private_manifest.read_bytes() != tracked_manifest.read_bytes():
        raise ValueError("Tracked and private model-freeze manifests differ")
    bundle = model_io.load_model_bundle(model_dir)
    manifest = bundle.manifest
    expected_hashes = {
        "raw_data_sha256": model_io._sha256_file(raw),
        "config_canonical_sha256": model_io._canonical_hash(config),
        "category_map_sha256": model_io._sha256_file(category_path),
    }
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported freeze manifest format")
    if manifest.get("source", {}).get("git_dirty") is not False:
        raise ValueError("Frozen models were fitted from a dirty source tree")
    if manifest.get("hashes") != expected_hashes:
        raise ValueError("Freeze manifest does not match the current data or configuration")
    expected_features = [
        *config["features"]["continuous"],
        *config["features"]["categorical"],
    ]
    if manifest.get("features") != {
        "all": expected_features,
        "categorical": list(config["features"]["categorical"]),
    }:
        raise ValueError("Frozen model features do not match the configuration")
    if float(manifest.get("propensity", -1)) != float(config["propensity"]):
        raise ValueError("Frozen propensity does not match the configuration")
    return bundle


def _sample_summary(frame: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "metric": "test_rows",
            "group": "all",
            "value": len(frame),
            "quantity_type": "sample_count",
            "unit": "rows",
        }
    ]
    for arm, label in ((0, "control"), (1, "treated")):
        arm_rows = frame["treatment"].eq(arm)
        rows.append(
            {
                "metric": "rows",
                "group": label,
                "value": int(arm_rows.sum()),
                "quantity_type": "sample_count",
                "unit": "rows",
            }
        )
        for outcome in outcomes:
            count = int(frame.loc[arm_rows, outcome].sum())
            rows.extend(
                [
                    {
                        "metric": f"{outcome}_events",
                        "group": label,
                        "value": count,
                        "quantity_type": "sample_count",
                        "unit": "events",
                    },
                    {
                        "metric": f"{outcome}_rate",
                        "group": label,
                        "value": float(frame.loc[arm_rows, outcome].mean()),
                        "quantity_type": "sample_rate",
                        "unit": "proportion",
                    },
                ]
            )
    result = pd.DataFrame(rows)
    result["sample_scope"] = "held_out_test"
    return result


def _ate_table(frame: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for outcome in outcomes:
        estimate = binary_ate(frame[outcome], frame["treatment"])
        rows.append(
            {
                "outcome": outcome,
                "estimate": estimate.estimate,
                "standard_error": estimate.standard_error,
                "ci_lower": estimate.ci_lower,
                "ci_upper": estimate.ci_upper,
                "n_treated": estimate.n_treated,
                "n_control": estimate.n_control,
                "quantity_type": "average_treatment_effect",
                "unit": "probability_difference",
                "sample_scope": "complete_source",
            }
        )
    return pd.DataFrame(rows)


def _policy_tables(
    summary: Sequence[Mapping[str, Any]], n_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(summary)
    frame["selected_rows"] = np.floor(frame["capacity"] * n_rows).astype("int64")
    frame["selected_fraction_exact"] = frame["selected_rows"] / n_rows
    frame["sample_scope"] = "held_out_test"
    frame["unit"] = "incremental_outcomes_per_test_row"
    frame["test_cohort_incremental_count_estimate"] = frame["estimate"] * n_rows
    frame["test_cohort_incremental_count_ci_lower"] = frame["ci_lower"] * n_rows
    frame["test_cohort_incremental_count_ci_upper"] = frame["ci_upper"] * n_rows
    frame["count_unit"] = "expected_incremental_outcomes_in_test_cohort"
    values = frame.loc[frame["estimand_type"].eq("policy_value")].copy()
    values["quantity_type"] = "held_out_policy_value"
    contrasts = frame.loc[frame["estimand_type"].eq("contrast")].copy()
    contrasts["quantity_type"] = "paired_policy_value_difference"
    return values, contrasts


def _qini_tables(
    outcomes: pd.DataFrame,
    scores: pd.DataFrame,
    propensity: float,
    tie_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = []
    curve_rows = []
    n_rows = len(outcomes)
    display_positions = np.unique(
        np.floor(np.linspace(0.0, 1.0, 101) * n_rows).astype(np.int64)
    )
    for policy in scores:
        curve = qini_curve(
            outcomes["conversion"],
            outcomes["treatment"],
            scores[policy],
            outcomes["row_id"],
            propensity=propensity,
            seed=tie_seed,
        )
        coefficients.append(
            {
                "policy": policy,
                "coefficient": curve.coefficient,
                "quantity_type": "qini_coefficient",
                "unit": "centered_ipw_gain_area_per_test_row",
                "sample_scope": "held_out_test",
            }
        )
        for position in display_positions:
            curve_rows.append(
                {
                    "policy": policy,
                    "fraction": curve.fraction[position],
                    "cumulative_gain": curve.cumulative_gain[position],
                    "centered_qini": curve.qini[position],
                    "unit": "incremental_conversions_per_test_row",
                    "sample_scope": "held_out_test",
                }
            )
    return pd.DataFrame(coefficients), pd.DataFrame(curve_rows)


def _plot_policy_values(values: pd.DataFrame, output: Path) -> list[Path]:
    plt = _pyplot()

    colors = {
        "random": "#777777",
        "response": "#D55E00",
        "t_learner": "#0072B2",
        "dr_learner": "#009E73",
    }
    outcomes = list(dict.fromkeys(values["outcome"]))
    figure, axes = plt.subplots(1, len(outcomes), figsize=(11.0, 4.2), squeeze=False)
    for axis, outcome in zip(axes[0], outcomes, strict=True):
        subset = values.loc[values["outcome"].eq(outcome)]
        for policy in colors:
            rows = subset.loc[subset["name"].eq(policy)].sort_values("capacity")
            x = rows["capacity"].to_numpy() * 100.0
            axis.plot(x, rows["estimate"], marker="o", label=policy, color=colors[policy])
            axis.fill_between(
                x,
                rows["ci_lower"],
                rows["ci_upper"],
                color=colors[policy],
                alpha=0.14,
            )
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.set_title(outcome.capitalize())
        axis.set_xlabel("Treatment capacity (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Incremental outcomes per test row")
    axes[0, -1].legend(frameon=False)
    figure.tight_layout()
    outputs = _save_figure(figure, output, pdf=True)
    plt.close(figure)
    return outputs


def _plot_contrasts(contrasts: pd.DataFrame, output: Path) -> list[Path]:
    plt = _pyplot()

    subset = contrasts.loc[contrasts["outcome"].eq("conversion")]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for name in dict.fromkeys(subset["name"]):
        rows = subset.loc[subset["name"].eq(name)].sort_values("capacity")
        x = rows["capacity"].to_numpy() * 100.0
        axis.plot(x, rows["estimate"], marker="o", label=name.replace("_", " "))
        axis.fill_between(x, rows["ci_lower"], rows["ci_upper"], alpha=0.12)
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set_xlabel("Treatment capacity (%)")
    axis.set_ylabel("Difference in conversions per test row")
    axis.set_title("Paired policy-value contrasts")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    outputs = _save_figure(figure, output, pdf=True)
    plt.close(figure)
    return outputs


def _plot_qini(curves: pd.DataFrame, output: Path) -> list[Path]:
    plt = _pyplot()

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for policy in dict.fromkeys(curves["policy"]):
        rows = curves.loc[curves["policy"].eq(policy)].sort_values("fraction")
        axis.plot(rows["fraction"] * 100.0, rows["centered_qini"], label=policy)
    axis.axhline(0.0, color="0.25", linewidth=0.8)
    axis.set_xlabel("Targeted fraction (%)")
    axis.set_ylabel("Centered IPW conversion gain per test row")
    axis.set_title("Held-out Qini curves")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    outputs = _save_figure(figure, output, pdf=True)
    plt.close(figure)
    return outputs


def run_evaluate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate frozen policies after enforcing the test-outcome access gate."""
    source_commit, source_dirty = model_io._git_state()
    if source_dirty:
        raise RuntimeError("Evaluation requires a clean source tree before outputs are written")
    preparation_manifest = _matching_preparation(config)
    bundle = _verify_freeze(config)
    _, processed, model_dir, results = _paths(config)
    feature_names = list(bundle.manifest["features"]["all"])
    feature_columns = ["row_id", *feature_names, "split"]

    # Test outcomes remain unread until every hash is verified and all frozen
    # score and nuisance predictions have been created from covariates alone.
    test_features = load_splits(config, ["test"], columns=feature_columns)
    test_features = apply_category_maps(
        test_features,
        _category_maps(model_dir / "category_map.json"),
        config["features"]["categorical"],
    )
    scores = bundle.predict_policy_scores(test_features)
    conversion_m1 = scores["response"].to_numpy(dtype=np.float64, copy=False)
    conversion_m0 = conversion_m1 - scores["t_learner"].to_numpy(
        dtype=np.float64, copy=False
    )
    visit_m0, visit_m1 = bundle.predict_nuisance(test_features, "visit")

    outcome_columns = [
        "row_id", "treatment", *config["features"]["outcomes"], "split"
    ]
    test_outcomes = load_splits(config, ["test"], columns=outcome_columns)
    if not np.array_equal(test_features["row_id"], test_outcomes["row_id"]):
        raise RuntimeError("Test feature and outcome rows are not aligned")

    treatment = test_outcomes["treatment"].to_numpy(dtype=np.int8, copy=False)
    propensity = float(config["propensity"])
    aipw_by_outcome = {
        "conversion": aipw_scores(
            test_outcomes["conversion"], treatment, conversion_m0, conversion_m1, propensity
        ),
        "visit": aipw_scores(
            test_outcomes["visit"], treatment, visit_m0, visit_m1, propensity
        ),
    }
    complete_outcomes = load_splits(
        config,
        ["train", "validation", "test"],
        columns=["row_id", "treatment", *config["features"]["outcomes"]],
    )
    bootstrap = paired_row_bootstrap(
        {name: scores[name].to_numpy(dtype=np.float64, copy=False) for name in scores},
        aipw_by_outcome,
        test_outcomes["row_id"],
        config["capacities"],
        replicates=int(config["bootstrap"]["replicates"]),
        seed=int(config["bootstrap"]["seed"]),
        tie_seed=int(config["seed"]),
    )

    n_rows = len(test_outcomes)
    sample = _sample_summary(test_outcomes, config["features"]["outcomes"])
    ate = _ate_table(complete_outcomes, config["features"]["outcomes"])
    policy_values, contrasts = _policy_tables(bootstrap.summary, n_rows)
    qini_coefficients, qini_curves = _qini_tables(
        test_outcomes, scores, propensity, int(config["seed"])
    )

    tables = results / "tables"
    figures = results / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    table_outputs = {
        "sample_summary": tables / "sample_summary.csv",
        "average_treatment_effects": tables / "average_treatment_effects.csv",
        "policy_values": tables / "policy_values.csv",
        "policy_contrasts": tables / "policy_contrasts.csv",
        "qini_coefficients": tables / "qini_coefficients.csv",
        "qini_curves": tables / "qini_curves.csv",
    }
    for frame, path in zip(
        (sample, ate, policy_values, contrasts, qini_coefficients, qini_curves),
        table_outputs.values(),
        strict=True,
    ):
        frame.to_csv(path, index=False)

    figure_outputs = [
        *_plot_policy_values(policy_values, figures / "policy_values"),
        *_plot_contrasts(contrasts, figures / "conversion_policy_contrasts"),
        *_plot_qini(qini_curves, figures / "qini_curves"),
    ]
    evaluation = {
        "evaluation_sample": "held_out_test",
        "test_rows": n_rows,
        "treatment_propensity": propensity,
        "capacities": [float(value) for value in config["capacities"]],
        "bootstrap": {
            "method": "paired_nonparametric_row_bootstrap",
            "replicates": int(config["bootstrap"]["replicates"]),
            "seed": int(config["bootstrap"]["seed"]),
            "interval": "pointwise_percentile_95_percent",
        },
        "policy_value_unit": "incremental_outcomes_per_test_row",
        "average_treatment_effect_sample": "complete_source",
        "test_cohort_count_definition": "policy_value multiplied by test_rows",
        "qini_curve_points_saved": int(qini_curves["fraction"].nunique()),
    }
    evaluation_path = results / "evaluation.json"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    output_paths = [*table_outputs.values(), *figure_outputs, evaluation_path]
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "evaluate",
        "integrity": {
            "freeze_verified_before_test_outcomes_read": True,
            "predictions_created_before_test_outcomes_read": True,
            "test_used_for_model_selection": False,
        },
        "inputs": {
            "prepared_manifest_sha256": model_io._sha256_file(
                processed / "_prepare_manifest.json"
            ),
            "freeze_manifest_sha256": model_io._sha256_file(
                model_dir / "freeze_manifest.json"
            ),
            "tracked_model_freeze_sha256": model_io._sha256_file(
                results / "manifests" / "model_freeze.json"
            ),
            **bundle.manifest["hashes"],
        },
        "source": {
            "git_commit_before_outputs": source_commit,
            "git_dirty_before_outputs": source_dirty,
        },
        "test_rows": n_rows,
        "outputs": {
            str(path.relative_to(results)): model_io._sha256_file(path)
            for path in output_paths
        },
        "prepared_rows": preparation_manifest["rows"],
    }
    run_manifest_path = results / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "test_rows": n_rows,
        "tables": [str(path) for path in table_outputs.values()],
        "figures": [str(path) for path in figure_outputs],
        "run_manifest": str(run_manifest_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uplift-policy", description="Reproduce the uplift-policy analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default="configs/analysis.yaml", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        output = run_prepare(config)
    elif args.command == "train":
        output = run_train(config)
    else:
        output = run_evaluate(config)
    print(json.dumps(output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
