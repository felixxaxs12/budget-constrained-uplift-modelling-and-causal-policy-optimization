# Budget-Constrained Uplift Modelling and Causal Policy Optimization

This repository studies treatment allocation under fixed capacity using `CRITEO-UPLIFTv2` and Criteo's official corrected `v2.1` artifact. The primary outcome is conversion. Visit is secondary.

The research design and official-data provenance checks are complete. No treatment-effect estimate, model result, policy value, analysis table, or analysis figure has been produced yet.

## Research question

On the official Criteo `CRITEO-UPLIFTv2` randomized benchmark, how much, if at all, can a policy learned from pre-treatment covariates improve held-out incremental conversions over random and response-based targeting at fixed, pre-specified treatment-capacity fractions?

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

The design review passed in two rounds. Phase 2 began only after the user confirmed the checkpoint.

## Phase 2 artifacts

- [Verified data manifest](data/manifest.json)
- [Data provenance and license](data/README.md)
- [Official downloader](scripts/download_data.py)
- [Targeted prior-art evidence](docs/prior_art.md)
- [Independent investigation review](docs/reviews/stage_02_investigation.md)

The raw 311,422,618-byte gzip is stored locally under `data/raw/` and is excluded from Git under Criteo's CC BY-NC-SA 4.0 terms. Its SHA-256 is `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`.

The Phase 2 review passed in two rounds. Phase 3 will begin only after the user confirms this checkpoint.

## Source

Criteo AI Lab reports 13,979,592 rows for the corrected release. This project independently verified that row count and the 16-column ordered header in the downloaded artifact.

- Dataset page: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Primary v2 paper: https://arxiv.org/abs/2111.10106
- Historical v1 paper: https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018
