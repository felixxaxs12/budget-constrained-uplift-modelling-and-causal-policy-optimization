# Reproducibility and statistical specification

## Estimand

The analysis targets intention to treat in the released-data distribution. For an evaluation cohort of size (n), score (s), capacity (q), and (k=\lfloor qn\rfloor), define the pre-specified tie breaker

\[
U_i=H(20260803,\operatorname{row\_id}_i).
\]

The batch rule (\Pi_{s,q}(X_{1:n},U_{1:n})) selects the first (k) rows ordered by decreasing (s(X_i)) and then increasing (U_i). Its incremental outcome value is

\[
\Delta_{Y,n}(\Pi_{s,q})=
E\left[\frac{1}{n}\sum_{i=1}^n
\pi_{s,q,i}\{Y_i(1)-Y_i(0)\}\right].
\]

The primary outcome is conversion and the secondary outcome is visit. Multiplying an estimate by the test-set size gives an expected incremental count for that held-out cohort, not realized conversions or campaign-wide impact.

## Data and split

The official gzip is read with an explicit schema. A zero-based `row_id` records source order and is never a feature. A pinned DuckDB hash of `row_id` and seed `20260803` assigns 60% of rows to train, 20% to validation, and 20% to test. Repeated anonymous records are retained because the dataset has no documented unique user identifier.

Before model fitting, the pipeline verifies the schema, finite feature values, binary domains, `treatment=0` implying `exposure=0`, and `conversion=1` implying `visit=1`. These checks do not estimate effects. Test treatment and outcomes are not used for model fitting, early stopping, score selection, or capacity selection.

Balance diagnostics use train and validation rows only. Continuous covariates use pooled-standard-deviation SMD. For each categorical covariate, the reported statistic is the maximum absolute one-versus-rest SMD across levels. The absolute 0.1 line is descriptive rather than a randomization test. A treatment classifier trained on train covariates and evaluated on validation covariates provides a joint diagnostic.

## Feature encoding

The four source-described continuous fields remain numeric. Levels of each categorical field are learned from train covariates, mapped to contiguous non-negative integers, and saved with the fitted artifacts. Levels absent from train are represented as missing. The same mapping is used for validation and test rows.

## Models

The treatment probability is fixed at the source design value (e_0=0.85). Outcome models use unweighted binary log-loss. No class weighting, outcome downsampling, calibration tournament, or hyperparameter sweep is used.

The frozen policy set is:

1. expected exact-(k) uniform random allocation;
2. response ranking by \(\widehat m_1^C(X)\);
3. T-learner ranking by \(\widehat m_1^C(X)-\widehat m_0^C(X)\); and
4. a DR-learner ranking score.

Arm-specific conversion and visit nuisance models select their boosting-round counts on validation log-loss. Visit models are used only to evaluate the conversion policies and do not define another policy.

For every development fold (f), conversion pseudo-outcomes are computed as

\[
\widetilde\tau_i=
\widehat m_{1,-f}^C(X_i)-\widehat m_{0,-f}^C(X_i)
+\frac{T_i}{e_0}\{C_i-\widehat m_{1,-f}^C(X_i)\}
-\frac{1-T_i}{1-e_0}\{C_i-\widehat m_{0,-f}^C(X_i)\}.
\]

Both nuisance predictions for row (i) come from models whose training fold excludes that row. The final DR regressor is fitted to three-fold out-of-fold pseudo-outcomes on the combined development data. Final arm-specific models are fitted on the same development rows with the previously selected round counts.

Before test outcomes are read, the pipeline records hashes of the raw file, configuration, category map, source commit, and fitted models together with all selected round counts and software versions.

## Average effects and held-out policy value

Full-sample average treatment effects are computed only after the model freeze:

\[
\widehat{ATE}_Y=\bar Y_1-\bar Y_0,
\qquad
\widehat{SE}(\widehat{ATE}_Y)=
\sqrt{\frac{s_1^2}{n_1}+\frac{s_0^2}{n_0}}.
\]

For outcome (Y\), its own development-fitted nuisance models produce the test-row score

\[
\widehat\psi_i^Y=
\widehat m_1^Y(X_i)-\widehat m_0^Y(X_i)
+\frac{T_i}{e_0}\{Y_i-\widehat m_1^Y(X_i)\}
-\frac{1-T_i}{1-e_0}\{Y_i-\widehat m_0^Y(X_i)\}.
\]

Policy value is

\[
\widehat\Delta_Y(\Pi_{s,q})=
\frac{1}{n}\sum_i\pi_{s,q,i}\widehat\psi_i^Y.
\]

The expected exact-(k) random baseline uses (p_q=k/n):

\[
\widehat\Delta_{Y,random}(q)=p_q\overline{\widehat\psi^Y}.
\]

The five pre-specified conversion contrasts at each capacity are response minus random, T-learner minus random, DR-learner minus random, T-learner minus response, and DR-learner minus response. At (q=1), every policy must have identical value; this is an implementation check rather than evidence of model performance.

## Ranking diagnostic

For conversion, define the fixed-propensity transformed outcome

\[
Z_i=\frac{T_iC_i}{e_0}-\frac{(1-T_i)C_i}{1-e_0}.
\]

For the ordering (r_s), the cumulative gain and centered Qini curve are

\[
G_s(j/n)=\frac{1}{n}\sum_{\ell=1}^j Z_{r_s(\ell)},
\qquad
Q_s(j/n)=G_s(j/n)-\frac{j}{n}G_s(1).
\]

The Qini coefficient is the trapezoidal area under (Q_s). A separate AUUC ranking is omitted because its area differs from the Qini coefficient by the same random-line area for every score on the same test data.

## Uncertainty

The analysis uses 1,000 paired nonparametric bootstrap replicates of test rows. One resampled count vector is shared by all policies and both outcomes in each replicate. Exact top-(k) membership is recomputed along each frozen score ordering, and paired differences are formed within the replicate before calculating standard errors and percentile intervals.

Intervals are pointwise over the five pre-specified capacities and condition on the fitted models. They cover row-IID test-sample and cutoff variation but not model refitting, advertiser clustering, temporal drift, or simultaneous inference across capacities.

## Commands

```bash
python scripts/download_data.py
uplift-policy prepare --config configs/analysis.yaml
uplift-policy train --config configs/analysis.yaml
uplift-policy evaluate --config configs/analysis.yaml
```

After `train`, commit the generated `results/manifests/model_freeze.json` before running `evaluate`. The evaluation command requires a clean working tree and verifies that this tracked file exactly matches the local freeze manifest. Generated aggregate artifacts are canonical; the paper and dashboard read them rather than reimplementing calculations.
