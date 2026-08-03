# Data provenance

## Release and retrieval

This project uses `CRITEO-UPLIFTv2` through Criteo's official `v2.1` artifact. Criteo describes this as the corrected or "un-biased" release. That wording is attributed to Criteo and does not mean this project has proved the absence of every possible bias.

- Artifact: `criteo-uplift-v2.1.csv.gz`
- Official redirect: https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz
- Resolved Criteo object: https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz
- Dataset page and terms: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Criteo-requested dataset citation: https://www.adkdd.org/papers/a-large-scale-benchmark-for-uplift-modeling/2018
- Primary v2 paper: https://arxiv.org/abs/2111.10106

| Check | Criteo-reported value | Locally observed value |
| --- | --- | --- |
| Compressed size | `297M` | 311,422,618 bytes |
| Data rows | 13,979,592 | 13,979,592 |
| Columns | `f0` through `f11`, plus four binary fields | exact ordered match |
| Average visit rate | 0.046992 | not computed in Phase 2 |
| Average conversion rate | 0.00292 | not computed in Phase 2 |
| Treatment ratio | 0.85 | not computed in Phase 2 |

Retrieval and integrity record:

- Retrieved at: `2026-08-03T20:15:15.309646+00:00`
- SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`
- Full gzip read and CRC check: passed
- Machine-readable record: [`manifest.json`](manifest.json)

## Field roles

The Criteo page's introductory paragraph says 11 features, but its field list and the 2021 v2 paper specify 12: `f0` through `f11`. The 2021 paper describes `f0`, `f2`, `f7`, and `f10` as continuous and the other eight features as categorical with anonymized modalities. The CSV stores all feature values numerically.

- `treatment`: source-described randomized assignment to the treatment population; this defines the intention-to-treat analysis.
- `conversion`: binary primary outcome.
- `visit`: binary secondary outcome.
- `exposure`: actual advertising exposure after assignment. It is not an estimated treatment effect and is excluded from model features.

The source documents imply two structural checks that will be recomputed in Phase 3: control rows should not be exposed, and a conversion should not occur without a visit.

## License and redistribution

Criteo applies the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license. The raw file remains under `data/raw/`, which Git ignores, and the repository does not mirror it.

Criteo-requested dataset citation:

> Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini. "A Large Scale Benchmark for Uplift Modeling." AdKDD and TargetAd, 2018.

Primary documentation for the corrected v2 release:

> Criteo Uplift Modeling Dataset, Criteo AI Lab. Released with Eustache Diemert, Artem Betlei, Christophe Renaudin, Massih-Reza Amini, Théophane Gregoir, and Thibaud Rahier, "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling," 2021.

License: https://creativecommons.org/licenses/by-nc-sa/4.0/

To reproduce the retrieval:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/download_data.py
```

No treatment-effect, model-performance, or policy-value result was computed in Phase 2.
