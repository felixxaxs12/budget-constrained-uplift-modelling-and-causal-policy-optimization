# Data provenance

This project uses `CRITEO-UPLIFTv2` through Criteo's official corrected `v2.1` artifact. Criteo describes the release as "un-biased"; this project attributes that wording to the publisher rather than claiming that every possible bias has been eliminated.

## Official artifact

- File: `criteo-uplift-v2.1.csv.gz`
- Dataset page: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Official redirect: https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
- Resolved Criteo object: https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz
- Locally verified size: 311,422,618 bytes
- Locally verified rows: 13,979,592
- SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- Full gzip read and CRC check: passed
- Machine-readable record: [manifest.json](manifest.json)

The ordered schema is `f0` through `f11`, followed by `treatment`, `conversion`, `visit`, and `exposure`. The source describes `f0`, `f2`, `f7`, and `f10` as continuous and the remaining feature columns as categorical. All feature tokens are stored numerically.

`treatment` is the randomized assignment used for the intention-to-treat analysis. `conversion` is the primary outcome and `visit` is secondary. `exposure` is measured after assignment and is used only for structural validation.

## Download

```bash
python scripts/download_data.py
```

The raw file is saved under `data/raw/`, which Git ignores. This repository chooses not to mirror the dataset; users obtain it directly from Criteo and are responsible for complying with its terms.

## Citation and license

Criteo requests citation of:

> Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini. "A Large Scale Benchmark for Uplift Modeling." AdKDD and TargetAd, 2018.

The corrected dataset design is documented in:

> Eustache Diemert, Artem Betlei, Christophe Renaudin, Massih-Reza Amini, Théophane Gregoir, and Thibaud Rahier. "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling." 2021.

Dataset license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).
