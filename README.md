# Budget-Constrained Uplift Modelling and Causal Policy Optimization

This repository studies treatment allocation under fixed capacity using the official unbiased Criteo Uplift v2.1 randomized dataset. The primary outcome is conversion. Visit is secondary.

The project is currently at the design-freeze stage. No dataset-derived estimate, model result, table, or figure has been produced yet.

## Research question

On the official unbiased Criteo Uplift v2.1 dataset, how much, if at all, can a policy learned from pre-treatment covariates improve held-out incremental conversions over random and response-based targeting at fixed, pre-specified treatment-capacity fractions?

## Fixed boundaries

- `treatment` is the randomized assignment and defines the intention-to-treat estimand.
- `exposure` occurs after assignment. It will not be used as a feature, ranking input, conditioning variable, or primary estimand.
- Capacity is the fraction of users who may be selected. The dataset has no monetary treatment cost or conversion value, so this project will not invent ROI or net-value results.
- The final test split will not be used for fitting, early stopping, model selection, or outcome-informed threshold tuning. Frozen scores on test covariates may be ranked to implement the pre-specified top-\(k\) capacity rule.
- Results will describe the released benchmark sample, not an actual campaign deployment.

## Stage 1 artifacts

- [Research protocol](docs/research_protocol.md)
- [Methodology blueprint](docs/methodology.md)
- [Prior-art and contribution boundary](docs/prior_art.md)
- [Minimal architecture](docs/architecture.md)
- [Data provenance and license](data/README.md)
- [Independent design review](docs/reviews/stage_01_scoping.md)

The implementation, downloaded data manifest, analyses, tables, figures, dashboard, and paper will be added only after the design review passes.

## Source

Criteo AI Lab describes the unbiased release as a 13,979,592-row CSV with 12 anonymized features, randomized treatment, two binary outcomes, and an exposure field. The official download is `criteo-uplift-v2.1.csv.gz`.

- Dataset page: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Benchmark paper: https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018
