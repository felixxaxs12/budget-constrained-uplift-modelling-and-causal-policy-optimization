# Budget-Constrained Uplift Modelling and Causal Policy Optimization

This repository evaluates whether causal targeting improves incremental conversions over response-based targeting when only a fixed fraction of users can receive treatment.

The study uses the official randomized `CRITEO-UPLIFTv2` dataset and its corrected `v2.1` artifact. It treats `treatment` as the randomized intention-to-treat assignment and excludes the post-assignment `exposure` field from every model.

## Method

Four allocation rules are compared at capacities of 5%, 10%, 20%, 50%, and 100%:

- expected uniform random allocation;
- response ranking by predicted conversion under treatment;
- a T-learner treatment-effect score; and
- a cross-fitted doubly robust learner.

Models are selected on train and validation data. The final comparison uses held-out augmented inverse-probability-weighted policy value, paired row bootstrap intervals, and an independently defined IPW Qini diagnostic. Capacity denotes equal-cost user slots, not a monetary budget.

## Reproduce

Python 3.11 through 3.13 is supported. LightGBM requires an OpenMP runtime; on macOS this is commonly installed with `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_data.py
uplift-policy prepare --config configs/analysis.yaml
uplift-policy train --config configs/analysis.yaml
# Commit the generated results/manifests/model_freeze.json, then continue
uplift-policy evaluate --config configs/analysis.yaml
pytest -q
```

The downloader verifies the official file's byte size, SHA-256 checksum, gzip integrity, row count, and ordered schema. See [data/README.md](data/README.md) for provenance and licensing.

Evaluation deliberately requires a clean working tree and an exact tracked copy of the model-freeze manifest. This makes the fitted-model hashes, training commit, selected boosting rounds, and software versions auditable before held-out outcome columns are loaded for evaluation or joined to frozen predictions. Model binaries remain local.

## Outputs

The analysis writes aggregate tables, figures, model and run manifests under `results/`. Row-level predictions, fitted model files, processed Parquet data, and the raw dataset remain outside Git. The Streamlit application reads the committed aggregate outputs and does not recompute statistics.

```bash
streamlit run app.py
```

The paper source and compiled manuscript are stored under `paper/`.

## Repository structure

```text
configs/analysis.yaml       Locked analysis settings
data/                       Provenance manifest and download instructions
docs/reproducibility.md     Estimands, model lifecycle, and evaluation definitions
scripts/download_data.py    Official-data downloader and integrity verification
src/uplift_policy/          Data, audit, model, evaluation, and orchestration code
tests/                      Algebraic and real-data integration checks
results/                    Canonical aggregate tables, figures, and manifests
paper/                      arXiv-style manuscript source and PDF
app.py                      Read-only results explorer
```

## Scope

Results describe offline policy evaluation within the released benchmark distribution. They are not estimates of advertiser ROI, realized campaign impact, exposure effects, or performance in a future deployment.

## License

Original code is released under the [MIT License](LICENSE). The Criteo dataset is not included in this repository and remains subject to Criteo's [CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/). The manuscript is released under CC BY 4.0.
