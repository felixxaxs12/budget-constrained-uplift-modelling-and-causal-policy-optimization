import importlib.util
from pathlib import Path

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


def test_streamlit_app_renders_without_exceptions() -> None:
    rendered = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not rendered.exception
    assert [tab.label for tab in rendered.tabs] == [
        "Conversions (primary)",
        "Visits (secondary)",
    ]
    assert len(rendered.dataframe) == 5
