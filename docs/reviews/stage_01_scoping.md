# Stage 1 design review

Date: 2026-08-03

Reviewer: independent reviewer subagent

Scope: research question, causal estimand, evaluation protocol, source boundaries, and architecture

## Round 1: REVISE

The first review found six blocking issues:

1. The individual-policy notation did not match batch top-`k` selection.
2. The role of test covariates in enforcing capacity conflicted with the README boundary on threshold selection.
3. The expected random baseline did not define the held-out sample or floor adjustment used in paired comparisons.
4. The bootstrap did not state whether it recomputed the capacity cutoff or what sources of uncertainty it covered.
5. The model objectives and lifecycle from validation through final nuisance fitting were not fully locked.
6. The draft assumed that institutional review would not apply.

The protocol and methodology were revised to address all six findings.

## Round 2: PASS

The reviewer confirmed that:

- the estimand now matches a batch top-`k` allocation rule;
- test-covariate ranking is separate from outcome-informed tuning;
- random and learned policies use the same held-out AIPW scores and realized `k/n`;
- each paired bootstrap replicate recomputes top-`k` selection while holding trained models fixed;
- classification and DR-regression objectives and the model lifecycle are specified; and
- the institutional statement is conditional on the policy applicable to the author.

The reviewer also found the planned architecture minimal and non-redundant. Each module has a separate responsibility, trivial boundaries may be merged during implementation, and speculative infrastructure and model families are excluded.

## Recorded cautions

- `row_id` may order exact score ties but must never enter a model feature matrix.
- The planned 200 bootstrap replicates give coarse percentile endpoints. The final paper must report this count and must not imply more numerical precision than the procedure supports.

No dataset result was computed during this review.
