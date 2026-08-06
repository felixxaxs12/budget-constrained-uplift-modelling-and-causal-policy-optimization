"""Build the manuscript tables from saved aggregate results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
CAPACITIES = (0.05, 0.10, 0.20, 0.50, 1.00)
POLICIES = ("random", "response", "t_learner", "dr_learner")
POLICY_LABELS = {
    "random": "Random",
    "response": "Treated-response",
    "t_learner": "T-learner",
    "dr_learner": "DR-learner",
}
OUTCOME_LABELS = {"conversion": "Conversion", "visit": "Visit"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / "tables" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(name: str, text: str) -> Path:
    path = TABLES / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def ci_cell(estimate: float, lower: float, upper: float, scale: float, digits: int = 1) -> str:
    template = f"{{:.{digits}f}} [{{:.{digits}f}}, {{:.{digits}f}}]"
    scaled = [estimate * scale, lower * scale, upper * scale]
    threshold = 0.5 * 10 ** (-digits)
    cleaned = [0.0 if abs(value) < threshold else value for value in scaled]
    return template.format(*cleaned)


def keyed(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate result row: {key}")
        result[key] = row
    return result


def build_sample_table() -> Path:
    rows = read_csv("sample_summary.csv")
    values = {(row["metric"], row["group"]): float(row["value"]) for row in rows}
    lines = [
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"Assignment & Rows & Conversions & Conversion rate (\%) & Visits & Visit rate (\%) \\",
        r"\midrule",
    ]
    for group, label in (("control", "Control"), ("treated", "Treated")):
        lines.append(
            f"{label} & {values[('rows', group)]:,.0f} & "
            f"{values[('conversion_events', group)]:,.0f} & "
            f"{100 * values[('conversion_rate', group)]:.4f} & "
            f"{values[('visit_events', group)]:,.0f} & "
            f"{100 * values[('visit_rate', group)]:.4f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return write("test_sample.tex", "\n".join(lines))


def build_ate_table() -> Path:
    rows = read_csv("average_treatment_effects.csv")
    if {row["sample_scope"] for row in rows} != {"complete_source"}:
        raise ValueError("ATE table must use the complete source sample")
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Outcome & Estimate (pp) & SE (pp) & 95\% CI (pp) & Treated / control $n$ \\",
        r"\midrule",
    ]
    for row in rows:
        estimate = 100 * float(row["estimate"])
        se = 100 * float(row["standard_error"])
        lower = 100 * float(row["ci_lower"])
        upper = 100 * float(row["ci_upper"])
        lines.append(
            f"{OUTCOME_LABELS[row['outcome']]} & {estimate:.4f} & {se:.4f} & "
            f"[{lower:.4f}, {upper:.4f}] & "
            f"{int(row['n_treated']):,} / {int(row['n_control']):,} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return write("full_sample_ate.tex", "\n".join(lines))


def build_estimator_scope_table() -> Path:
    ate_rows = read_csv("average_treatment_effects.csv")
    conversion_ate = next(row for row in ate_rows if row["outcome"] == "conversion")
    sample_rows = read_csv("sample_summary.csv")
    values = {(row["metric"], row["group"]): float(row["value"]) for row in sample_rows}
    test_difference = values[("conversion_rate", "treated")] - values[("conversion_rate", "control")]
    policy_rows = read_csv("policy_values.csv")
    full_policy = next(
        row
        for row in policy_rows
        if row["outcome"] == "conversion"
        and row["capacity"] == "1.0"
        and row["name"] == "response"
    )
    lines = [
        r"\begin{tabular}{@{}lll@{}}",
        r"\toprule",
        r"Quantity & Sample and estimator & Incremental conversions per 100,000 \\",
        r"\midrule",
        (
            "Average treatment effect & Complete source, difference in means & "
            + ci_cell(
                float(conversion_ate["estimate"]),
                float(conversion_ate["ci_lower"]),
                float(conversion_ate["ci_upper"]),
                100_000,
            )
            + r" \\"
        ),
        (
            "Raw assignment difference & Held-out test, difference in means & "
            f"{100_000 * test_difference:.1f} (no interval reported) \\\\"
        ),
        (
            "All-treatment policy value & Held-out test, AIPW & "
            + ci_cell(
                float(full_policy["estimate"]),
                float(full_policy["ci_lower"]),
                float(full_policy["ci_upper"]),
                100_000,
            )
            + r" \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return write("estimator_scope.tex", "\n".join(lines))


def build_policy_value_table() -> Path:
    rows = read_csv("policy_values.csv")
    index = keyed(rows, ("outcome", "capacity", "name"))
    lines = [
        r"\begin{tabular}{@{}rllll@{}}",
        r"\toprule",
        r"Capacity (\%) & Random & Treated-response & T-learner & DR-learner \\",
        r"\midrule",
    ]
    for capacity in CAPACITIES:
        cells = []
        for policy in POLICIES:
            row = index[("conversion", str(capacity), policy)]
            if row["sample_scope"] != "held_out_test":
                raise ValueError("Policy values must use the held-out test sample")
            cells.append(
                ci_cell(
                    float(row["estimate"]),
                    float(row["ci_lower"]),
                    float(row["ci_upper"]),
                    100_000,
                )
            )
        lines.append(f"{100 * capacity:.0f} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return write("conversion_policy_values.tex", "\n".join(lines))


def transformed_contrast(
    index: dict[tuple[str, ...], dict[str, str]],
    outcome: str,
    capacity: float,
    name: str,
    reverse: bool = False,
) -> tuple[float, float, float]:
    row = index[(outcome, str(capacity), name)]
    estimate = float(row["estimate"])
    lower = float(row["ci_lower"])
    upper = float(row["ci_upper"])
    if reverse:
        return -estimate, -upper, -lower
    return estimate, lower, upper


def build_contrast_table(outcome: str, file_name: str) -> Path:
    rows = read_csv("policy_contrasts.csv")
    index = keyed(rows, ("outcome", "capacity", "name"))
    lines = [
        r"\begin{tabular}{@{}rlll@{}}",
        r"\toprule",
        r"Capacity (\%) & Response $-$ random & Response $-$ T-learner & Response $-$ DR-learner \\",
        r"\midrule",
    ]
    for capacity in CAPACITIES:
        contrasts = (
            transformed_contrast(index, outcome, capacity, "response_minus_random"),
            transformed_contrast(index, outcome, capacity, "t_learner_minus_response", reverse=True),
            transformed_contrast(index, outcome, capacity, "dr_learner_minus_response", reverse=True),
        )
        cells = [ci_cell(*contrast, scale=100_000) for contrast in contrasts]
        lines.append(f"{100 * capacity:.0f} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return write(file_name, "\n".join(lines))


def build_qini_table() -> Path:
    rows = read_csv("qini_coefficients.csv")
    index = {row["policy"]: row for row in rows}
    if set(index) != set(POLICIES[1:]):
        raise ValueError("Unexpected Qini policy set")
    lines = [
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"Ranking score & Qini coefficient ($\times 10^{4}$) \\",
        r"\midrule",
    ]
    for policy in POLICIES[1:]:
        lines.append(f"{POLICY_LABELS[policy]} & {10_000 * float(index[policy]['coefficient']):.3f} \\\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return write("qini_coefficients.tex", "\n".join(lines))


def build_model_table() -> Path:
    metadata_path = RESULTS / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = {
        "conversion_m0": r"Conversion $\widehat m_0$",
        "conversion_m1": r"Conversion $\widehat m_1$",
        "visit_m0": r"Visit $\widehat m_0$",
        "visit_m1": r"Visit $\widehat m_1$",
        "dr_learner": "DR-learner",
    }
    lines = [
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        r"Model & Selected rounds & Validation loss \\",
        r"\midrule",
    ]
    for key in ("conversion_m0", "conversion_m1", "visit_m0", "visit_m1", "dr_learner"):
        lines.append(
            f"{labels[key]} & {metadata['selected_rounds'][key]} & "
            f"{metadata['validation_losses'][key]:.6f} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return write("model_selection.tex", "\n".join(lines))


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    for builder in (
        build_sample_table(),
        build_ate_table(),
        build_estimator_scope_table(),
        build_policy_value_table(),
        build_contrast_table("conversion", "conversion_contrasts.tex"),
        build_contrast_table("visit", "visit_contrasts.tex"),
        build_qini_table(),
        build_model_table(),
    ):
        print(builder.relative_to(ROOT))


if __name__ == "__main__":
    main()
