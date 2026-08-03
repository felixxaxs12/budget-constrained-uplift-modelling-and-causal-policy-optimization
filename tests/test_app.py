import importlib.util
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SPEC = importlib.util.spec_from_file_location("results_app", APP_PATH)
assert SPEC is not None and SPEC.loader is not None
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


def test_selected_capacity_uses_committed_aggregate_results() -> None:
    results = app.load_results()
    selected = app.rows_at_capacity(results["policy"], "conversion", 0.2)

    assert set(selected["name"]) == set(app.POLICY_LABELS)
    assert selected["selected_rows"].nunique() == 1
    assert int(selected["selected_rows"].iloc[0]) == 559_418

    display = app.policy_table(results["policy"], "conversion", 0.2)
    assert display["Allocation rule"].tolist() == list(app.POLICY_LABELS.values())


def test_hypothetical_scenario_is_separate_linear_calculation() -> None:
    policy_values = app.load_results()["policy"]
    row = app.rows_at_capacity(policy_values, "conversion", 0.1).loc[
        lambda frame: frame["name"] == "response"
    ].iloc[0]
    scenario = app.hypothetical_economics(
        row, value_per_incremental_conversion=1.0, cost_per_user=0.0
    )

    expected_value = row["test_cohort_incremental_count_estimate"]
    expected_cost = 0.0
    assert np.isclose(scenario["incremental_value"], expected_value)
    assert np.isclose(scenario["treatment_cost"], expected_cost)
    assert np.isclose(scenario["net_value"], expected_value - expected_cost)


def test_app_has_no_model_or_row_level_data_dependency() -> None:
    source = Path(app.__file__).read_text(encoding="utf-8")

    assert "uplift_policy" not in source
    assert "lightgbm" not in source.lower()
    assert "data/raw" not in source
    assert "data/processed" not in source


def test_streamlit_app_renders_without_exceptions() -> None:
    rendered = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not rendered.exception
    assert [tab.label for tab in rendered.tabs] == [
        "Conversions (primary)",
        "Visits (secondary)",
    ]
    assert len(rendered.dataframe) == 5
