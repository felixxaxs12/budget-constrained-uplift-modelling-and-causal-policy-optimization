"""Read-only explorer for the study's committed aggregate results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

CAPACITY_OPTIONS = {
    "5%": 0.05,
    "10%": 0.10,
    "20%": 0.20,
    "50%": 0.50,
    "100%": 1.00,
}
POLICY_LABELS = {
    "random": "Expected random allocation",
    "response": "Response targeting",
    "t_learner": "T-learner uplift targeting",
    "dr_learner": "Doubly robust uplift targeting",
}
CONTRAST_LABELS = {
    "response_minus_random": "Response targeting minus random",
    "t_learner_minus_random": "T-learner minus random",
    "dr_learner_minus_random": "Doubly robust learner minus random",
    "t_learner_minus_response": "T-learner minus response targeting",
    "dr_learner_minus_response": "Doubly robust learner minus response targeting",
}
POLICY_ORDER = {name: order for order, name in enumerate(POLICY_LABELS)}
CONTRAST_ORDER = {name: order for order, name in enumerate(CONTRAST_LABELS)}


@st.cache_data(show_spinner=False)
def load_results() -> dict[str, pd.DataFrame]:
    """Load only committed, aggregate analysis tables."""

    return {
        "ate": pd.read_csv(TABLE_DIR / "average_treatment_effects.csv"),
        "policy": pd.read_csv(TABLE_DIR / "policy_values.csv"),
        "contrast": pd.read_csv(TABLE_DIR / "policy_contrasts.csv"),
        "qini": pd.read_csv(TABLE_DIR / "qini_coefficients.csv"),
        "sample": pd.read_csv(TABLE_DIR / "sample_summary.csv"),
    }


def rows_at_capacity(
    frame: pd.DataFrame, outcome: str, capacity: float
) -> pd.DataFrame:
    """Return result rows for one outcome and capacity."""

    mask = (frame["outcome"] == outcome) & np.isclose(
        frame["capacity"].to_numpy(), capacity
    )
    return frame.loc[mask].copy()


def policy_table(
    policy_values: pd.DataFrame, outcome: str, capacity: float
) -> pd.DataFrame:
    """Prepare a compact, reader-facing policy value table."""

    selected = rows_at_capacity(policy_values, outcome, capacity)
    selected["_order"] = selected["name"].map(POLICY_ORDER)
    selected = selected.sort_values("_order")
    return pd.DataFrame(
        {
            "Allocation rule": selected["name"].map(POLICY_LABELS),
            "Selected users": selected["selected_rows"].map(
                lambda value: f"{int(value):,}"
            ),
            "Incremental outcomes per 100,000 test users": selected["estimate"]
            * 100_000,
            "Pointwise 95% CI per 100,000": [
                f"{low * 100_000:,.1f} to {high * 100_000:,.1f}"
                for low, high in zip(selected["ci_lower"], selected["ci_upper"])
            ],
            "Estimated incremental outcomes in test cohort": selected[
                "test_cohort_incremental_count_estimate"
            ],
            "Pointwise 95% CI for cohort count": [
                f"{low:,.1f} to {high:,.1f}"
                for low, high in zip(
                    selected["test_cohort_incremental_count_ci_lower"],
                    selected["test_cohort_incremental_count_ci_upper"],
                )
            ],
        }
    )


def contrast_table(
    contrasts: pd.DataFrame, outcome: str, capacity: float
) -> pd.DataFrame:
    """Prepare paired policy comparisons for display."""

    selected = rows_at_capacity(contrasts, outcome, capacity)
    selected["_order"] = selected["name"].map(CONTRAST_ORDER)
    selected = selected.sort_values("_order")
    return pd.DataFrame(
        {
            "Comparison": selected["name"].map(CONTRAST_LABELS),
            "Difference per 100,000 test users": selected["estimate"] * 100_000,
            "Pointwise 95% paired-bootstrap CI": [
                f"{low * 100_000:,.1f} to {high * 100_000:,.1f}"
                for low, high in zip(selected["ci_lower"], selected["ci_upper"])
            ],
            "Estimated difference in test-cohort outcomes": selected[
                "test_cohort_incremental_count_estimate"
            ],
            "Pointwise 95% CI for cohort difference": [
                f"{low:,.1f} to {high:,.1f}"
                for low, high in zip(
                    selected["test_cohort_incremental_count_ci_lower"],
                    selected["test_cohort_incremental_count_ci_upper"],
                )
            ],
        }
    )


def hypothetical_economics(
    policy_row: pd.Series, value_per_incremental_conversion: float, cost_per_user: float
) -> dict[str, float]:
    """Combine fixed study estimates with user-supplied economic assumptions."""

    treatment_cost = float(policy_row["selected_rows"]) * cost_per_user
    return {
        "incremental_value": float(
            policy_row["test_cohort_incremental_count_estimate"]
        )
        * value_per_incremental_conversion,
        "treatment_cost": treatment_cost,
        "net_value": float(policy_row["test_cohort_incremental_count_estimate"])
        * value_per_incremental_conversion
        - treatment_cost,
        "net_value_lower": float(
            policy_row["test_cohort_incremental_count_ci_lower"]
        )
        * value_per_incremental_conversion
        - treatment_cost,
        "net_value_upper": float(
            policy_row["test_cohort_incremental_count_ci_upper"]
        )
        * value_per_incremental_conversion
        - treatment_cost,
    }


def render_outcome(
    results: dict[str, pd.DataFrame], outcome: str, capacity: float
) -> None:
    noun = "conversions" if outcome == "conversion" else "visits"
    title = "Primary outcome: conversions" if outcome == "conversion" else "Secondary outcome: visits"
    st.subheader(title)
    st.caption(
        f"Held-out policy value estimates the incremental {noun} attributable to each "
        "allocation rule. Intervals are pointwise 95% row-bootstrap intervals based on "
        "1,000 resamples; the same resamples are used for policy comparisons."
    )

    selected = rows_at_capacity(results["policy"], outcome, capacity)
    highest = selected.loc[selected["estimate"].idxmax()]
    tied_for_highest = np.isclose(selected["estimate"], highest["estimate"])
    highest_label = (
        "All rules tie"
        if int(tied_for_highest.sum()) == len(selected)
        else POLICY_LABELS[highest["name"]]
    )
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Treatment slots", f"{int(highest['selected_rows']):,}")
    metric_b.metric("Highest point estimate", highest_label)
    metric_c.metric(
        f"Estimated incremental {noun}",
        f"{highest['test_cohort_incremental_count_estimate']:,.1f}",
    )
    st.caption(
        "“Highest point estimate” is descriptive of this held-out evaluation and is not "
        "a guarantee of future performance."
    )

    display = policy_table(results["policy"], outcome, capacity)
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Incremental outcomes per 100,000 test users": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "Estimated incremental outcomes in test cohort": st.column_config.NumberColumn(
                format="%.1f"
            ),
        },
    )

    st.markdown("#### Paired comparisons")
    st.caption(
        "A positive difference favors the rule named first. These are offline estimates, "
        "not observed differences from deploying the rules in separate campaigns."
    )
    comparisons = contrast_table(results["contrast"], outcome, capacity)
    st.dataframe(
        comparisons,
        hide_index=True,
        width="stretch",
        column_config={
            "Difference per 100,000 test users": st.column_config.NumberColumn(
                format="%.1f"
            ),
            "Estimated difference in test-cohort outcomes": st.column_config.NumberColumn(
                format="%.1f"
            ),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="Causal Targeting Results",
        page_icon="◉",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    results = load_results()
    test_rows = int(
        results["sample"].loc[
            (results["sample"]["metric"] == "test_rows")
            & (results["sample"]["group"] == "all"),
            "value",
        ].iloc[0]
    )

    st.title("Capacity-Constrained Causal Targeting")
    st.write(
        "Explore the locked, held-out results from the randomized Criteo uplift benchmark. "
        "The primary question is how four allocation rules compare when treatment can be "
        "assigned to only a fixed share of users."
    )
    st.info(
        "Capacity is the share of users who can receive treatment under equal per-user "
        "slots. It is not a monetary budget. Results are offline causal estimates for this "
        "benchmark, not realized campaign impact or advertiser ROI."
    )
    st.caption(
        f"Fixed held-out test cohort: {test_rows:,} users · 1,000 bootstrap resamples · "
        "Read-only aggregate results"
    )

    capacity_label = st.select_slider(
        "Treatment capacity",
        options=list(CAPACITY_OPTIONS),
        value="20%",
        help="The largest share of the fixed test cohort that the policy may select.",
    )
    capacity = CAPACITY_OPTIONS[capacity_label]
    st.caption(
        f"Selected capacity: {capacity_label} of the held-out cohort. Every rule receives "
        "the same number of treatment slots."
    )

    conversion_tab, visit_tab = st.tabs(["Conversions (primary)", "Visits (secondary)"])
    with conversion_tab:
        render_outcome(results, "conversion", capacity)
    with visit_tab:
        render_outcome(results, "visit", capacity)

    st.divider()
    st.header("Experiment-wide context")
    st.caption(
        "Average treatment effects use all 13,979,592 randomized records. Policy values "
        "above use only the fixed held-out test cohort, so they answer a different question."
    )
    ate_conversion = results["ate"].loc[
        results["ate"]["outcome"] == "conversion"
    ].iloc[0]
    ate_visit = results["ate"].loc[results["ate"]["outcome"] == "visit"].iloc[0]
    ate_left, ate_right = st.columns(2)
    ate_left.metric("Conversion ATE", f"{ate_conversion['estimate'] * 100:.3f} pp")
    ate_left.caption(
        f"95% analytic CI: {ate_conversion['ci_lower'] * 100:.3f} to "
        f"{ate_conversion['ci_upper'] * 100:.3f} percentage points"
    )
    ate_right.metric("Visit ATE", f"{ate_visit['estimate'] * 100:.3f} pp")
    ate_right.caption(
        f"95% analytic CI: {ate_visit['ci_lower'] * 100:.3f} to "
        f"{ate_visit['ci_upper'] * 100:.3f} percentage points"
    )

    st.markdown("#### Capacity frontier")
    st.image(
        str(FIGURE_DIR / "policy_values.png"),
        caption=(
            "Held-out AIPW policy values with pointwise 95% row-bootstrap intervals. "
            "Conversion is the primary outcome; visit is secondary."
        ),
        width="stretch",
    )

    st.markdown("#### Qini ranking diagnostic")
    st.caption(
        "The centered IPW Qini curve assesses conversion ranking over the full targeting "
        "depth. Its area is a ranking diagnostic, not a policy-value or ROI estimate."
    )
    qini = results["qini"].copy()
    qini["_order"] = qini["policy"].map(POLICY_ORDER)
    qini = qini.sort_values("_order")
    qini_display = pd.DataFrame(
        {
            "Ranking rule": qini["policy"].map(POLICY_LABELS),
            "Centered IPW Qini area × 100,000": qini["coefficient"] * 100_000,
        }
    )
    qini_left, qini_right = st.columns([1, 2])
    with qini_left:
        st.dataframe(
            qini_display,
            hide_index=True,
            width="stretch",
            column_config={
                "Centered IPW Qini area × 100,000": st.column_config.NumberColumn(
                    format="%.2f"
                )
            },
        )
    with qini_right:
        st.image(
            str(FIGURE_DIR / "qini_curves.png"),
            caption="Centered conversion Qini curves on the held-out test cohort.",
            width="stretch",
        )

    st.divider()
    with st.expander("Optional hypothetical economic scenario"):
        st.warning(
            "Criteo does not provide conversion value or treatment cost. Inputs here are "
            "user-supplied assumptions. This calculator does not change the study results "
            "and does not estimate realized ROI."
        )
        scenario_policy = st.selectbox(
            "Allocation rule",
            options=list(POLICY_LABELS),
            format_func=POLICY_LABELS.get,
        )
        scenario_left, scenario_right = st.columns(2)
        with scenario_left:
            conversion_value = st.number_input(
                "Hypothetical value per incremental conversion",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Enter any consistent currency unit.",
            )
        with scenario_right:
            treatment_cost = st.number_input(
                "Hypothetical treatment cost per selected user",
                min_value=0.0,
                value=0.0,
                step=0.01,
                help="Use the same currency unit as conversion value.",
            )

        selected_policy = rows_at_capacity(
            results["policy"], "conversion", capacity
        ).loc[lambda frame: frame["name"] == scenario_policy].iloc[0]
        scenario = hypothetical_economics(
            selected_policy, conversion_value, treatment_cost
        )
        if conversion_value == 0.0 and treatment_cost == 0.0:
            st.caption("Enter assumptions above to calculate a hypothetical scenario.")
        scenario_a, scenario_b, scenario_c = st.columns(3)
        scenario_a.metric(
            "Value from estimated incremental conversions",
            f"{scenario['incremental_value']:,.2f}",
        )
        scenario_b.metric(
            "Treatment cost for selected users", f"{scenario['treatment_cost']:,.2f}"
        )
        scenario_c.metric("Hypothetical net value", f"{scenario['net_value']:,.2f}")
        st.caption(
            "Pointwise interval implied by the conversion-count interval and fixed user "
            f"inputs: {scenario['net_value_lower']:,.2f} to "
            f"{scenario['net_value_upper']:,.2f} currency units. It does not include "
            "uncertainty in the user-supplied economic assumptions."
        )

    st.divider()
    st.caption(
        "Source: committed aggregate outputs in results/. The application does not load "
        "row-level data, fitted models, or test-set predictions."
    )


if __name__ == "__main__":
    main()
