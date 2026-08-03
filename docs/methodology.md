---
origin_skill: deep-research
origin_mode: full
origin_date: 2026-08-03
verification_status: design_only
version_label: methodology_v1
---

# Methodology blueprint

## Design

This is a quantitative secondary analysis of `CRITEO-UPLIFTv2`, using Criteo's official `v2.1` artifact. The causal target is intention to treat. The study recruits no participants and introduces no new intervention. Before dissemination, the author will confirm any institutional determination required by the policy that applies to the author. The public-data terms and attribution requirements apply.

The study follows a locked train, validation, and test design. Before the model freeze, test rows are accessed only for pre-specified schema and structural-integrity checks that do not estimate rates or effects. Test outcome summaries, average treatment effects, ranking diagnostics, and policy values remain unopened until estimator tests, model fitting, and validation-stage decisions are complete. The full-sample average-effect estimates are produced during the same single final-results stage, after every model and reporting decision is frozen.

## Data and split

1. Download `criteo-uplift-v2.1.csv.gz` from Criteo's official link.
2. Record the URL, download time, byte size, and SHA-256 checksum.
3. Ingest the compressed CSV with an explicit schema and add a stable `row_id`. `row_id` is never a feature.
4. Validate column names, numeric finiteness, binary fields, row count, and the two source-defined structural relations: control implies no exposure, and conversion implies visit.
5. Do not drop repeated feature or outcome rows. The dataset has no documented unique user identifier, so repeated anonymous records are not proven duplicates.
6. Assign rows reproducibly to 60% train, 20% validation, and 20% test using a seeded hash of `row_id` with seed `20260803`.

The 2021 v2 paper classifies `f0`, `f2`, `f7`, and `f10` as continuous and the remaining eight features as categorical, although all values are stored numerically. Phase 3 will validate those storage properties against the downloaded file before creating one typed feature matrix.

Raw and processed data remain outside Git. The repository stores the downloader, checksum manifest, schema report, split manifest, and aggregate results.

## Experiment audit

- Report treatment and control counts and the observed assignment fraction.
- Compute standardized mean differences for `f0` through `f11` and plot them against a descriptive absolute 0.1 warning line. This line is not proof of randomization.
- Fit one treatment-classification diagnostic on train and evaluate AUC on validation to detect joint predictability that marginal SMDs can miss. No p-value balance table will be added merely because the sample is large.
- Confirm that `exposure` is excluded from every design matrix.

## Average effects

The estimators and their algebraic tests are implemented before model fitting, but no real-data average treatment effect is computed or reported until the final-results stage. At that point the pre-specified estimate uses all released rows. This full-sample estimand describes the released randomized benchmark and is separate from the held-out comparison of learned policies.

For each binary outcome \(Y\in\{C,V\}\), estimate

\[
\widehat{ATE}=\bar Y_1-\bar Y_0
\]

with the unpooled difference-in-means standard error

\[
\widehat{SE}(\widehat{ATE})=
\sqrt{\frac{s_1^2}{n_1}+\frac{s_0^2}{n_0}}.
\]

Report arm counts, arm rates, the absolute risk difference, and a 95% confidence interval. Relative uplift is descriptive and will not replace the absolute effect.

## Locked policy set

All learned policies use the same typed feature matrix and one documented set of tree and regularization settings. LightGBM receives the four source-described continuous columns as numeric and the other eight as categorical. The outcome models use binary log-loss, while the DR second-stage model uses squared-error loss because its target is continuous. There is no parallel one-hot representation and no broad hyperparameter sweep.

1. Expected random allocation: allocate exactly \(k=\lfloor qn_{test}\rfloor\) slots uniformly without replacement. Evaluation integrates over that random allocation, so no arbitrary draw is needed.
2. Response ranking: rank by \(\widehat m_1(X)=\widehat P(C=1\mid X,T=1)\). After validation freezes its boosting-round count, this model is refitted on treated train and validation rows and reused by the T-learner.
3. T-learner: rank by \(\widehat m_1(X)-\widehat m_0(X)\).
4. DR-learner: form three-fold cross-fitted AIPW pseudo-outcomes on train using a constant treatment propensity estimated from the training assignment rate, then regress the pseudo-outcome on \(X\) to obtain a ranking score.

No S-learner, X-learner, causal forest, neural network, or ensemble is included unless a later reviewer identifies a specific unanswered question that requires it.

### Model lifecycle

1. Fit arm-specific outcome models on train and use the corresponding validation arm's binary log-loss to choose the boosting-round count for each arm.
2. With those counts fixed, create three-fold out-of-fold nuisance predictions on train and form the DR training pseudo-outcome. Fit the DR second-stage regressor on train; choose its boosting-round count by squared error against validation pseudo-outcomes computed from train-fitted nuisance models.
3. Freeze all boosting-round counts and other model settings. Estimate the constant assignment propensity from combined train and validation rows. Recreate cross-fitted pseudo-outcomes on these combined development data, then fit the final DR second-stage model on them. Fit the final arm-specific nuisance models on the same development data.
4. Use the final \(\widehat m_1\) for response scores, \(\widehat m_1-\widehat m_0\) for T-learner scores, and the final DR regressor for DR scores. The same final nuisance models produce the independent test-set AIPW scores. No model is fitted or selected using test assignments or outcomes.

## Held-out policy evaluation

For test-row nuisance predictions \(\widehat m_1(X_i)\), \(\widehat m_0(X_i)\), and the assignment rate \(\widehat e\) estimated on combined train and validation rows, compute

\[
\widehat\psi_i=
\widehat m_1(X_i)-\widehat m_0(X_i)
+\frac{T_i}{\widehat e}\{Y_i-\widehat m_1(X_i)\}
-\frac{1-T_i}{1-\widehat e}\{Y_i-\widehat m_0(X_i)\}.
\]

For a frozen-score batch rule \(\Pi_q\), let \(k=\lfloor qn_{test}\rfloor\) and select the \(k\) highest test scores. Estimate incremental value by

\[
\widehat\Delta(\Pi_q)=\frac{1}{n_{test}}\sum_i\pi_{qi}(X_{1:n_{test}})\widehat\psi_i.
\]

The expected uniform-random baseline uses \(p_q=k/n_{test}\):

\[
\widehat\Delta_{random}(q)=p_q\overline{\widehat\psi},
\qquad
\widehat\Delta(\Pi_q)-\widehat\Delta_{random}(q)
=\frac{1}{n_{test}}\sum_i(\pi_{qi}-p_q)\widehat\psi_i.
\]

Each learned score uses a deterministic hash of `row_id` to order exact ties. The identifier is not a model feature and has no role outside tie-breaking and reproducible splitting. Computing the pre-specified top-\(k\) cutoff from frozen test-feature scores is part of enforcing capacity; it is not threshold tuning. Test assignments and outcomes remain unavailable to this step.

The primary uncertainty procedure is a 200-replicate paired nonparametric bootstrap of test rows. Within each replicate, every policy uses the same resampled row indices, top-\(k\) selection is recomputed from its frozen scores, and the random baseline uses \(k/n\). Report paired bootstrap standard errors and 95% percentile intervals. The interval therefore covers held-out-row sampling and cutoff variation with the trained models held fixed. It does not measure instability from refitting models or changing validation decisions. Conventional uplift/Qini curves and AUUC are reported as ranking diagnostics, not as substitutes for policy value.

The test set is used once for the final comparison. Its covariates may implement the pre-specified batch capacity rule, but its assignments and outcomes are not used for fitting, early stopping, model selection, or threshold tuning. No policy is relabeled as primary after seeing its test result.

## Secondary analysis

The locked conversion policies are also evaluated on `visit`. This does not introduce a second model tournament. It asks whether the allocation chosen for conversion has a different effect on the secondary outcome.

## Validity and limitations

| Issue | Handling |
| --- | --- |
| Released sample was non-uniformly subsampled | Limit inference to the released benchmark and avoid recovering original campaign effects. |
| Treatment arm is much larger than control | Use the documented randomized design, report counts, and quantify uncertainty. Do not rebalance the estimand. |
| Conversion is rare | Report confidence intervals and paired comparisons; do not infer individual-effect accuracy. |
| Advertiser and time identifiers are absent | Use an IID held-out split and state that external and temporal validity are untested. |
| Exposure is post-assignment | Exclude it from features and primary estimands. |
| Test-set adaptivity | Freeze the policy set, capacity grid, estimators, and reporting rules before opening test results. |

## Reproducibility rules

- One analysis configuration holds seeds, split fractions, capacity grid, model settings, and output paths.
- Every table and figure is generated by the analysis pipeline from the official file.
- Unit-level algebra checks may use scalar examples, but no synthetic dataset or simulated result will appear in the analysis or paper. Integration tests use the official data.
- Failed stages stop with the actual error. The pipeline will not silently retry with a different estimator or data subset.
- Data, model, and result manifests record hashes and software versions.

## Design-freeze decision

Primary decision: `sound`, subject to independent review before implementation. The main open risk is empirical precision at low capacities, not a mismatch between the question and the randomized design.
