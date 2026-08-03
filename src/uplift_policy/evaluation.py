"""Statistical evaluation for the locked uplift-policy protocol."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_PROPENSITY = 0.85
DEFAULT_TIE_SEED = 20260803
SCORE_POLICIES = ("response", "t_learner", "dr_learner")
LOCKED_CONTRASTS = (
    ("response_minus_random", "response", "random"),
    ("t_learner_minus_random", "t_learner", "random"),
    ("dr_learner_minus_random", "dr_learner", "random"),
    ("t_learner_minus_response", "t_learner", "response"),
    ("dr_learner_minus_response", "dr_learner", "response"),
)

_UINT64_MASK = (1 << 64) - 1
_GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)


@dataclass(frozen=True)
class ATEEstimate:
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    n_treated: int
    n_control: int


@dataclass(frozen=True)
class QiniCurve:
    fraction: NDArray[np.float64]
    cumulative_gain: NDArray[np.float64]
    qini: NDArray[np.float64]
    coefficient: float


@dataclass(frozen=True)
class PairedBootstrapResult:
    capacities: NDArray[np.float64]
    point_policy_values: dict[str, dict[str, NDArray[np.float64]]]
    point_contrasts: dict[str, dict[str, NDArray[np.float64]]]
    summary: tuple[dict[str, str | int | float], ...]
    bootstrap_policy_values: (
        dict[str, dict[str, NDArray[np.float64]]] | None
    ) = None
    bootstrap_contrasts: (
        dict[str, dict[str, NDArray[np.float64]]] | None
    ) = None


def _aligned_1d(*arrays: ArrayLike) -> tuple[NDArray, ...]:
    converted = tuple(np.asarray(array) for array in arrays)
    if not converted or any(array.ndim != 1 for array in converted):
        raise ValueError("inputs must be one-dimensional")
    if len({array.size for array in converted}) != 1:
        raise ValueError("inputs must have the same length")
    return converted


def aipw_scores(
    outcome: ArrayLike,
    treatment: ArrayLike,
    m0: ArrayLike,
    m1: ArrayLike,
    propensity: float = DEFAULT_PROPENSITY,
) -> NDArray[np.float64]:
    """Return row-level fixed-propensity AIPW treatment-effect scores."""
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must lie strictly between zero and one")
    y, t, mu0, mu1 = _aligned_1d(outcome, treatment, m0, m1)
    y = y.astype(np.float64, copy=False)
    t = t.astype(np.float64, copy=False)
    mu0 = mu0.astype(np.float64, copy=False)
    mu1 = mu1.astype(np.float64, copy=False)
    return (
        mu1
        - mu0
        + t * (y - mu1) / propensity
        - (1.0 - t) * (y - mu0) / (1.0 - propensity)
    )


def binary_ate(
    outcome: ArrayLike,
    treatment: ArrayLike,
    confidence: float = 0.95,
) -> ATEEstimate:
    """Difference in binary means with the locked analytic standard error."""
    y, t = _aligned_1d(outcome, treatment)
    y = y.astype(np.float64, copy=False)
    if not np.isin(y, (0.0, 1.0)).all() or not np.isin(t, (0, 1)).all():
        raise ValueError("outcome and treatment must be binary")
    t = t.astype(np.int8, copy=False)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    treated = y[t == 1]
    control = y[t == 0]
    if treated.size < 2 or control.size < 2:
        raise ValueError("each treatment arm needs at least two rows")

    estimate = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    )
    critical_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return ATEEstimate(
        estimate=estimate,
        standard_error=standard_error,
        ci_lower=estimate - critical_value * standard_error,
        ci_upper=estimate + critical_value * standard_error,
        n_treated=int(treated.size),
        n_control=int(control.size),
    )


def tie_keys(
    row_id: ArrayLike, seed: int = DEFAULT_TIE_SEED
) -> NDArray[np.uint64]:
    """Map non-negative row identifiers to deterministic SplitMix64 keys."""
    ids = np.asarray(row_id)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("row_id must be a one-dimensional integer array")
    if np.any(ids < 0) or not 0 <= seed <= _UINT64_MASK:
        raise ValueError("row_id and seed must be unsigned 64-bit values")

    mixed = ids.astype(np.uint64, copy=False) + np.uint64(seed) + _GOLDEN_GAMMA
    mixed = (mixed ^ (mixed >> 30)) * _MIX_1
    mixed = (mixed ^ (mixed >> 27)) * _MIX_2
    return mixed ^ (mixed >> 31)


def policy_order(
    scores: ArrayLike,
    row_id: ArrayLike,
    seed: int = DEFAULT_TIE_SEED,
) -> NDArray[np.int64]:
    """Order rows by decreasing score and then by the fixed tie key."""
    score, ids = _aligned_1d(scores, row_id)
    score = score.astype(np.float64, copy=False)
    if not np.isfinite(score).all():
        raise ValueError("policy scores must be finite")
    keys = tie_keys(ids, seed)
    # row_id is a deterministic tertiary key for the negligible hash-collision case.
    return np.lexsort((ids, keys, -score)).astype(np.int64, copy=False)


def capacity_count(n_rows: int, capacity: float) -> int:
    if n_rows < 1 or not 0.0 <= capacity <= 1.0:
        raise ValueError("capacity must be in [0, 1] and n_rows must be positive")
    return int(np.floor(float(capacity) * n_rows))


def top_k_membership(
    scores: ArrayLike,
    row_id: ArrayLike,
    capacity: float,
    seed: int = DEFAULT_TIE_SEED,
) -> NDArray[np.bool_]:
    """Return exact membership for the frozen batch top-k policy."""
    order = policy_order(scores, row_id, seed)
    selected = np.zeros(order.size, dtype=np.bool_)
    selected[order[: capacity_count(order.size, capacity)]] = True
    return selected


def policy_value(
    aipw: ArrayLike,
    scores: ArrayLike,
    row_id: ArrayLike,
    capacity: float,
    seed: int = DEFAULT_TIE_SEED,
) -> float:
    psi, score, ids = _aligned_1d(aipw, scores, row_id)
    membership = top_k_membership(score, ids, capacity, seed)
    return float(np.dot(membership, psi.astype(np.float64, copy=False)) / psi.size)


def expected_random_value(aipw: ArrayLike, capacity: float) -> float:
    """Expected value of exact-k uniform random allocation."""
    psi = np.asarray(aipw, dtype=np.float64)
    if psi.ndim != 1 or psi.size == 0:
        raise ValueError("aipw must be a non-empty one-dimensional array")
    return capacity_count(psi.size, capacity) / psi.size * float(psi.mean())


def locked_contrasts(
    policy_values: Mapping[str, float | NDArray[np.float64]],
) -> dict[str, float | NDArray[np.float64]]:
    """Form the five pre-specified paired policy contrasts."""
    return {
        name: policy_values[left] - policy_values[right]
        for name, left, right in LOCKED_CONTRASTS
    }


def qini_curve(
    conversion: ArrayLike,
    treatment: ArrayLike,
    scores: ArrayLike,
    row_id: ArrayLike,
    propensity: float = DEFAULT_PROPENSITY,
    seed: int = DEFAULT_TIE_SEED,
) -> QiniCurve:
    """Compute the protocol's IPW cumulative gain and centered Qini curve."""
    if not 0.0 < propensity < 1.0:
        raise ValueError("propensity must lie strictly between zero and one")
    y, t, score, ids = _aligned_1d(conversion, treatment, scores, row_id)
    y = y.astype(np.float64, copy=False)
    t = t.astype(np.float64, copy=False)
    transformed = t * y / propensity - (1.0 - t) * y / (1.0 - propensity)
    order = policy_order(score, ids, seed)
    n_rows = y.size
    fraction = np.arange(n_rows + 1, dtype=np.float64) / n_rows
    gain = np.empty(n_rows + 1, dtype=np.float64)
    gain[0] = 0.0
    gain[1:] = np.cumsum(transformed[order], dtype=np.float64) / n_rows
    qini = gain - fraction * gain[-1]
    coefficient = float(
        np.sum((qini[:-1] + qini[1:]) * np.diff(fraction) / 2.0)
    )
    return QiniCurve(fraction, gain, qini, coefficient)


def bootstrap_count_vectors(
    n_rows: int, replicates: int, seed: int
) -> Iterator[NDArray[np.int64]]:
    """Yield exact multinomial row-bootstrap counts, one vector at a time."""
    if n_rows < 1 or replicates < 1:
        raise ValueError("n_rows and replicates must be positive")
    generator = np.random.default_rng(seed)
    for _ in range(replicates):
        sampled_rows = generator.integers(0, n_rows, size=n_rows, dtype=np.int64)
        yield np.bincount(sampled_rows, minlength=n_rows).astype(np.int64, copy=False)


def _weighted_top_k_sums_ordered(
    ordered_counts: NDArray[np.int64],
    ordered_values: NDArray[np.float64],
    k_values: NDArray[np.int64],
    total_values: NDArray[np.float64],
    total_count: int,
) -> NDArray[np.float64]:
    result = np.zeros((k_values.size, ordered_values.shape[1]), dtype=np.float64)
    full = k_values == total_count
    result[full] = total_values

    active_positions = np.flatnonzero((k_values > 0) & ~full)
    if active_positions.size == 0:
        return result

    cumulative_counts = np.cumsum(ordered_counts, dtype=np.int64)
    boundaries = np.searchsorted(
        cumulative_counts, k_values[active_positions], side="left"
    )
    last = int(boundaries.max()) + 1
    cumulative_values = np.cumsum(
        ordered_counts[:last, None] * ordered_values[:last], axis=0
    )
    for output_position, boundary in zip(active_positions, boundaries, strict=True):
        before_count = cumulative_counts[boundary - 1] if boundary else 0
        before_value = cumulative_values[boundary - 1] if boundary else 0.0
        remainder = k_values[output_position] - before_count
        result[output_position] = before_value + remainder * ordered_values[boundary]
    return result


def weighted_top_k_sums(
    counts: ArrayLike,
    order: ArrayLike,
    values: ArrayLike,
    k_values: ArrayLike,
) -> NDArray[np.float64]:
    """Sum values for exact top-k slots in a row-bootstrap sample."""
    count, ranking = _aligned_1d(counts, order)
    count = count.astype(np.int64, copy=False)
    ranking = ranking.astype(np.int64, copy=False)
    value_matrix = np.asarray(values, dtype=np.float64)
    if value_matrix.ndim == 1:
        value_matrix = value_matrix[:, None]
    if value_matrix.ndim != 2 or value_matrix.shape[0] != count.size:
        raise ValueError("values must have one row per count")
    if (
        np.any(count < 0)
        or np.any(ranking < 0)
        or np.any(ranking >= count.size)
    ):
        raise ValueError("counts and order contain invalid values")
    ks = np.asarray(k_values, dtype=np.int64)
    if ks.ndim != 1 or np.any(ks < 0) or np.any(ks > count.sum()):
        raise ValueError("k_values must lie between zero and the total count")
    total_values = count @ value_matrix
    return _weighted_top_k_sums_ordered(
        count[ranking], value_matrix[ranking], ks, total_values, int(count.sum())
    )


def percentile_interval(
    bootstrap_samples: ArrayLike, confidence: float = 0.95, axis: int = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    samples = np.asarray(bootstrap_samples, dtype=np.float64)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, (tail, 1.0 - tail), axis=axis)
    return lower, upper


def paired_row_bootstrap(
    scores: Mapping[str, ArrayLike],
    aipw_by_outcome: Mapping[str, ArrayLike],
    row_id: ArrayLike,
    capacities: Sequence[float],
    *,
    replicates: int = 1000,
    seed: int = 20260805,
    tie_seed: int = DEFAULT_TIE_SEED,
    confidence: float = 0.95,
    keep_replicates: bool = False,
) -> PairedBootstrapResult:
    """Evaluate frozen policies with one paired row bootstrap for all outcomes."""
    if set(scores) != set(SCORE_POLICIES):
        raise ValueError(f"scores must contain exactly {SCORE_POLICIES}")
    if not aipw_by_outcome:
        raise ValueError("at least one outcome is required")
    if replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")

    ids = np.asarray(row_id)
    outcome_names = tuple(aipw_by_outcome)
    value_matrix = np.column_stack(
        [np.asarray(aipw_by_outcome[name], dtype=np.float64) for name in outcome_names]
    )
    if ids.ndim != 1 or value_matrix.shape[0] != ids.size or ids.size == 0:
        raise ValueError("row_id and outcome scores must have the same non-zero length")

    n_rows = ids.size
    capacity_array = np.asarray(capacities, dtype=np.float64)
    if capacity_array.ndim != 1 or capacity_array.size == 0:
        raise ValueError("capacities must be a non-empty sequence")
    k_values = np.array(
        [capacity_count(n_rows, capacity) for capacity in capacity_array],
        dtype=np.int64,
    )

    orders = {
        policy: policy_order(scores[policy], ids, tie_seed) for policy in SCORE_POLICIES
    }
    ordered_values = {policy: value_matrix[order] for policy, order in orders.items()}
    policy_names = ("random",) + SCORE_POLICIES
    point_by_policy: dict[str, NDArray[np.float64]] = {}
    total_values = value_matrix.sum(axis=0)
    point_by_policy["random"] = (
        k_values[:, None] / n_rows * total_values[None, :] / n_rows
    )
    unit_counts = np.ones(n_rows, dtype=np.int64)
    for policy in SCORE_POLICIES:
        point_by_policy[policy] = _weighted_top_k_sums_ordered(
            unit_counts[orders[policy]],
            ordered_values[policy],
            k_values,
            total_values,
            n_rows,
        ) / n_rows

    bootstrap_by_policy = {
        policy: np.empty(
            (replicates, capacity_array.size, len(outcome_names)), dtype=np.float64
        )
        for policy in policy_names
    }
    count_vectors = bootstrap_count_vectors(n_rows, replicates, seed)
    for replicate, counts in enumerate(count_vectors):
        replicate_totals = counts @ value_matrix
        bootstrap_by_policy["random"][replicate] = (
            k_values[:, None] / n_rows * replicate_totals[None, :] / n_rows
        )
        for policy in SCORE_POLICIES:
            bootstrap_by_policy[policy][replicate] = _weighted_top_k_sums_ordered(
                counts[orders[policy]],
                ordered_values[policy],
                k_values,
                replicate_totals,
                n_rows,
            ) / n_rows

    full_positions = np.flatnonzero(k_values == n_rows)
    for position in full_positions:
        point_reference = point_by_policy["random"][position]
        if any(
            not np.array_equal(point_by_policy[policy][position], point_reference)
            for policy in SCORE_POLICIES
        ):
            raise AssertionError("all point estimates must agree at full capacity")
        reference = bootstrap_by_policy["random"][:, position]
        if any(
            not np.array_equal(bootstrap_by_policy[policy][:, position], reference)
            for policy in SCORE_POLICIES
        ):
            raise AssertionError("all policies must agree at full capacity")

    point_values = {
        outcome: {
            policy: point_by_policy[policy][:, outcome_index].copy()
            for policy in policy_names
        }
        for outcome_index, outcome in enumerate(outcome_names)
    }
    replicate_values = {
        outcome: {
            policy: bootstrap_by_policy[policy][:, :, outcome_index].copy()
            for policy in policy_names
        }
        for outcome_index, outcome in enumerate(outcome_names)
    }
    point_contrasts = {
        outcome: locked_contrasts(values) for outcome, values in point_values.items()
    }
    replicate_contrasts = {
        outcome: locked_contrasts(values)
        for outcome, values in replicate_values.items()
    }

    summary_rows: list[dict[str, str | int | float]] = []
    for outcome in outcome_names:
        series_groups = (
            ("policy_value", point_values[outcome], replicate_values[outcome]),
            ("contrast", point_contrasts[outcome], replicate_contrasts[outcome]),
        )
        for estimand_type, estimates, samples in series_groups:
            for name, estimate in estimates.items():
                standard_error = samples[name].std(axis=0, ddof=1)
                lower, upper = percentile_interval(samples[name], confidence, axis=0)
                for capacity_index, capacity in enumerate(capacity_array):
                    summary_rows.append(
                        {
                            "outcome": outcome,
                            "capacity": float(capacity),
                            "estimand_type": estimand_type,
                            "name": name,
                            "estimate": float(estimate[capacity_index]),
                            "bootstrap_standard_error": float(
                                standard_error[capacity_index]
                            ),
                            "ci_lower": float(lower[capacity_index]),
                            "ci_upper": float(upper[capacity_index]),
                            "bootstrap_replicates": replicates,
                        }
                    )

    return PairedBootstrapResult(
        capacities=capacity_array,
        point_policy_values=point_values,
        point_contrasts=point_contrasts,
        summary=tuple(summary_rows),
        bootstrap_policy_values=replicate_values if keep_replicates else None,
        bootstrap_contrasts=replicate_contrasts if keep_replicates else None,
    )
