---
origin_skill: deep-research
origin_mode: full
origin_date: 2026-08-03
verification_status: design_only
version_label: research_v1
---

# Research question brief

## Topic

Capacity-constrained causal targeting on `CRITEO-UPLIFTv2` using Criteo's official corrected `v2.1` artifact.

## Primary research question

On the official Criteo `CRITEO-UPLIFTv2` randomized benchmark, how much, if at all, can a policy learned from pre-treatment covariates improve held-out incremental conversions over random and response-based targeting at fixed, pre-specified treatment-capacity fractions?

## Primary estimand

The decision rule is a batch allocation rule. For an evaluation cohort of size \(n\), a frozen score ranks the cohort's feature vectors \(X_{1:n}\), and \(\Pi_q(X_{1:n})=(\pi_{q1},\ldots,\pi_{qn})\) selects exactly \(k=\lfloor qn\rfloor\) rows. The incremental conversion value is

\[
\Delta_{C,n}(\Pi_q)=
E\left[\frac{1}{n}\sum_{i=1}^n
\pi_{qi}(X_{1:n})\{C_i(1)-C_i(0)\}\right].
\]

The expectation is over evaluation cohorts from the released benchmark population and their potential outcomes. The rule may use the evaluation cohort's covariates to enforce the pre-specified capacity, but it may not use its treatment assignments or observed outcomes. Multiplying an estimate by the held-out sample size gives an incremental count for that held-out sample only.

The primary comparisons are paired differences in \(\Delta_{C,n}\) between learned allocation rules and two baselines at the same capacity: expected uniform random allocation of exactly \(k\) rows and response-based ranking.

## FINER assessment

| Criterion | Score | Reason |
| --- | ---: | --- |
| Feasible | 5/5 | The public randomized data contain assignment, outcomes, and pre-treatment covariates. |
| Interesting | 5/5 | The question separates raw response prediction from treatment-induced response. |
| Novel | 3/5 | Budgeted uplift is established. The intended contribution is an open, uncertainty-aware replication and evaluation, not a new estimator. |
| Ethical | 5/5 | The study uses an anonymized public dataset and creates no new intervention. License compliance is still required. |
| Relevant | 5/5 | The estimand directly represents allocation under a fixed treatment capacity. |
| Average | 4.6/5 | No criterion falls below the ARS threshold. |

## Scope

### In scope

- Population: the rows in the downloaded v2.1 file.
- Features: `f0` through `f11` only.
- Assignment: `treatment`, analyzed as intention to treat.
- Primary outcome: `conversion`.
- Secondary outcome: `visit`.
- Capacity: equal-cost user slots at \(q\in\{0.05,0.10,0.20,0.50,1.00\}\).
- Policies: expected random allocation, response ranking, a T-learner, and a cross-fitted DR-learner.
- Evaluation: a locked test set and policy-value uncertainty at every capacity.
- `exposure`: structural validation only, including checking that exposed rows belong to the treatment arm.

### Out of scope

- Monetary ROI, net value, ad price, or conversion value.
- Per-protocol, complier, mediation, or exposure-conditioned effects.
- Individual-effect truth or PEHE, because both potential outcomes are not observed.
- Online bandits, pacing, dynamic treatment, or temporal deployment claims.
- Advertiser, geographic, or demographic heterogeneity, because those identifiers are unavailable.
- Generalization beyond the released mixture of incrementality tests.

## Assumptions

- Consistency and no interference between rows.
- Positive probability of treatment and control under the randomized assignment.
- The documented features were measured before assignment.
- The deterministic split represents the released benchmark population.
- Policy scores use features only. The allocation rule uses those frozen scores, the evaluation cohort's features, and the pre-specified capacity. It never uses evaluation assignments, outcomes, or exposure. A hash of `row_id` orders exact score ties only; `row_id` is not a model feature.
- Each selected row consumes one unit of capacity.

## Subquestions and bindings

1. What are the conversion and visit average treatment effects and their uncertainty in the released sample?
   - Inherits the full population, assignment, feature, and generalization boundaries.
2. At each fixed capacity, how do the four locked policies compare in held-out incremental conversions and paired uncertainty?
   - Inherits the full scope with conversion as the primary outcome.
3. Do conventional ranking metrics and held-out incremental policy value lead to the same model ordering?
   - Inherits the full scope. This is an evaluation comparison, not a claim that either metric observes individual effects.

## Evidence status

The official `v2.1` artifact has been downloaded and its byte size, SHA-256, gzip integrity, ordered header, and 13,979,592 data rows have been verified. The two-week outcome window and feature timing remain source-described properties from the 2021 paper. Treatment, outcome, effect, model, and policy-value statistics have not yet been computed.

## Reporting boundary

The paper will describe the work as a reproducible empirical benchmark or replication unless the literature review and completed results support a narrower original contribution. A higher projected score, incremental count, or policy value will not be called realized business impact.
