"""Algebraic tests for evaluation code; these fixtures are not research data."""

from __future__ import annotations

import numpy as np
import pytest

from uplift_policy.evaluation import (
    POLICY_CONTRASTS,
    aipw_scores,
    binary_ate,
    bootstrap_count_vectors,
    expected_random_value,
    policy_contrasts,
    paired_row_bootstrap,
    percentile_interval,
    policy_order,
    policy_value,
    qini_curve,
    tie_keys,
    top_k_membership,
    weighted_top_k_sums,
)


def _splitmix64_reference(row_id: int, seed: int) -> int:
    mask = (1 << 64) - 1
    value = (row_id + seed + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def test_aipw_scores_follow_fixed_propensity_formula() -> None:
    result = aipw_scores(
        outcome=np.array([1.0, 0.0]),
        treatment=np.array([1, 0]),
        m0=np.array([0.2, 0.1]),
        m1=np.array([0.8, 0.7]),
        propensity=0.8,
    )
    np.testing.assert_allclose(result, np.array([0.85, 1.10]))


def test_binary_ate_uses_sample_variances_by_arm() -> None:
    estimate = binary_ate(
        outcome=np.array([1, 0, 1, 0, 0, 1]),
        treatment=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert estimate.estimate == pytest.approx(1.0 / 3.0)
    assert estimate.standard_error == pytest.approx(np.sqrt(2.0) / 3.0)
    assert estimate.n_treated == 3
    assert estimate.n_control == 3
    assert estimate.ci_lower < estimate.estimate < estimate.ci_upper


def test_tie_keys_match_scalar_splitmix64_reference() -> None:
    ids = np.array([0, 1, 2, 2**32], dtype=np.int64)
    expected = np.array(
        [_splitmix64_reference(int(row_id), 20260803) for row_id in ids],
        dtype=np.uint64,
    )
    np.testing.assert_array_equal(tie_keys(ids), expected)
    np.testing.assert_array_equal(tie_keys(ids), tie_keys(ids))
    assert not np.array_equal(tie_keys(ids), tie_keys(ids, seed=20260804))


def test_policy_order_and_membership_use_exact_hashed_tie_break() -> None:
    ids = np.arange(6, dtype=np.int64)
    scores = np.array([0.9, 0.8, 0.8, 0.2, 0.1, 0.0])
    keys = np.array([_splitmix64_reference(int(row), 20260803) for row in ids])
    tied_order = [1, 2] if keys[1] < keys[2] else [2, 1]
    expected_order = np.array([0, *tied_order, 3, 4, 5])
    np.testing.assert_array_equal(policy_order(scores, ids), expected_order)

    selected = top_k_membership(scores, ids, capacity=0.5)
    assert selected.sum() == 3
    np.testing.assert_array_equal(np.flatnonzero(selected), np.sort(expected_order[:3]))


def test_policy_and_expected_random_values_use_floor_k_over_n() -> None:
    psi = np.array([1.0, 2.0, 3.0, 4.0])
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    ids = np.arange(4)
    assert policy_value(psi, scores, ids, 0.5) == pytest.approx(0.75)
    assert expected_random_value(psi, 0.5) == pytest.approx(1.25)
    assert expected_random_value(psi, 0.49) == pytest.approx(0.625)


def test_policy_contrasts_have_the_five_prespecified_comparisons() -> None:
    values = {
        "random": 1.0,
        "response": 2.0,
        "t_learner": 4.0,
        "dr_learner": 7.0,
    }
    contrasts = policy_contrasts(values)
    assert tuple(contrasts) == tuple(item[0] for item in POLICY_CONTRASTS)
    assert contrasts == {
        "response_minus_random": 1.0,
        "t_learner_minus_random": 3.0,
        "dr_learner_minus_random": 6.0,
        "t_learner_minus_response": 2.0,
        "dr_learner_minus_response": 5.0,
    }


def test_qini_curve_matches_hand_calculated_ipw_example() -> None:
    result = qini_curve(
        conversion=np.array([1, 1, 0, 0]),
        treatment=np.array([1, 0, 1, 0]),
        scores=np.array([4.0, 1.0, 3.0, 2.0]),
        row_id=np.arange(4),
        propensity=0.5,
    )
    np.testing.assert_allclose(result.fraction, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(result.cumulative_gain, [0.0, 0.5, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(result.qini, result.cumulative_gain)
    assert result.coefficient == pytest.approx(0.375)
    assert result.qini[-1] == 0.0


def test_bootstrap_counts_are_exact_and_reproducible() -> None:
    first = list(bootstrap_count_vectors(7, 3, seed=19))
    second = list(bootstrap_count_vectors(7, 3, seed=19))
    assert all(counts.sum() == 7 for counts in first)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)


def test_weighted_top_k_sums_handle_boundary_row_copies_exactly() -> None:
    counts = np.array([2, 0, 1, 1])
    order = np.array([2, 0, 1, 3])
    values = np.column_stack(
        [np.array([1.0, 2.0, 3.0, 4.0]), np.array([10.0, 20.0, 30.0, 40.0])]
    )
    result = weighted_top_k_sums(counts, order, values, np.arange(5))
    np.testing.assert_allclose(result[:, 0], [0.0, 3.0, 4.0, 5.0, 9.0])
    np.testing.assert_allclose(result[:, 1], [0.0, 30.0, 40.0, 50.0, 90.0])


def test_paired_bootstrap_shares_counts_and_enforces_full_capacity() -> None:
    ids = np.arange(8)
    conversion = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0])
    scores = {
        "response": np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float),
        "t_learner": np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float),
        "dr_learner": np.array([1, 8, 2, 7, 3, 6, 4, 5], dtype=float),
    }
    result = paired_row_bootstrap(
        scores=scores,
        aipw_by_outcome={"conversion": conversion, "visit": 2.0 * conversion},
        row_id=ids,
        capacities=(0.25, 0.5, 1.0),
        replicates=40,
        seed=31,
        keep_replicates=True,
    )

    assert result.bootstrap_policy_values is not None
    assert result.bootstrap_contrasts is not None
    for policy in ("random", "response", "t_learner", "dr_learner"):
        conversion_samples = result.bootstrap_policy_values["conversion"][policy]
        visit_samples = result.bootstrap_policy_values["visit"][policy]
        np.testing.assert_allclose(visit_samples, 2.0 * conversion_samples)
        np.testing.assert_array_equal(
            conversion_samples[:, -1],
            result.bootstrap_policy_values["conversion"]["random"][:, -1],
        )
        assert result.point_policy_values["conversion"][policy][-1] == (
            result.point_policy_values["conversion"]["random"][-1]
        )

    for contrast in (item[0] for item in POLICY_CONTRASTS):
        np.testing.assert_array_equal(
            result.bootstrap_contrasts["conversion"][contrast][:, -1],
            np.zeros(40),
        )
        np.testing.assert_allclose(
            result.bootstrap_contrasts["visit"][contrast],
            2.0 * result.bootstrap_contrasts["conversion"][contrast],
        )

    assert len(result.summary) == 2 * 3 * (4 + 5)
    assert set(result.summary[0]) == {
        "outcome",
        "capacity",
        "estimand_type",
        "name",
        "estimate",
        "bootstrap_standard_error",
        "ci_lower",
        "ci_upper",
        "bootstrap_replicates",
    }


def test_paired_bootstrap_is_reproducible_without_returning_replicates() -> None:
    ids = np.arange(5)
    scores = {
        "response": np.arange(5, dtype=float),
        "t_learner": -np.arange(5, dtype=float),
        "dr_learner": np.array([1.0, 3.0, 2.0, 5.0, 4.0]),
    }
    arguments = dict(
        scores=scores,
        aipw_by_outcome={"conversion": np.arange(5, dtype=float)},
        row_id=ids,
        capacities=(0.2, 1.0),
        replicates=12,
        seed=7,
    )
    first = paired_row_bootstrap(**arguments)
    second = paired_row_bootstrap(**arguments)
    assert first.summary == second.summary
    assert first.bootstrap_policy_values is None
    assert first.bootstrap_contrasts is None


def test_percentile_interval_uses_two_sided_quantiles() -> None:
    lower, upper = percentile_interval(np.array([0.0, 1.0, 2.0, 3.0]), 0.5)
    assert float(lower) == pytest.approx(0.75)
    assert float(upper) == pytest.approx(2.25)
