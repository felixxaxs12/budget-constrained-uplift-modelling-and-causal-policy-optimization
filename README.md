# Capacity-Constrained Uplift Modelling on Criteo

This project asks a practical targeting question: if an advertiser can treat only a fixed share of eligible users, which ranking rule produces the most incremental conversions?

I tested four allocation rules on 13,979,592 rows from the corrected CRITEO-UPLIFTv2.1 randomized advertising dataset. The analysis constrains the fraction of users treated; it does not optimize a monetary budget.

[Paper](paper/main.pdf) | [Results](results/) | [Analysis code](src/uplift_policy/) | [Dashboard](#dashboard)

## Main result

Treated-response ranking performed better than the two estimated treatment-effect rankings in this experiment. At 10% capacity, held-out conversion policy values were:

| Allocation rule | Estimated incremental conversions per 100,000 eligible users |
|---|---:|
| Expected random allocation | 9.8 |
| Treated-response ranking | 87.1 |
| T-learner ranking | 78.1 |
| Doubly robust learner ranking | 57.4 |

The paired difference between treated-response and random allocation was 77.3 conversions per 100,000 (pointwise 95% interval: 65.1 to 90.0). Treated-response also exceeded the T-learner by 9.0 (1.8 to 15.3) and the doubly robust learner by 29.7 (21.8 to 37.5).

Treated-response had the highest estimated conversion value at 5%, 10%, 20%, and 50% capacity. At 100%, every rule treats the same users and therefore has the same value. The complete-data conversion average treatment effect was 0.1152 percentage points (95% confidence interval: 0.1085 to 0.1219).

![Held-out policy values across treatment capacities](results/figures/policy_values.png)

These are offline estimates for the released benchmark. They are not realized campaign outcomes or return-on-investment estimates.

## What I did

- Downloaded the dataset from Criteo and checked its byte size, checksum, schema, row count, and gzip integrity.
- Treated randomized assignment as the intervention and excluded `exposure`, which occurs after assignment.
- Used a fixed 60/20/20 train, validation, and test split. Test outcomes did not enter model selection or ranking construction.
- Compared expected random allocation, predicted response under treatment, a LightGBM T-learner, and a three-fold cross-fitted doubly robust learner.
- Evaluated exact top-capacity policies with held-out augmented inverse-probability-weighted value estimates and 1,000 paired row-bootstrap replicates.
- Reported Qini curves as a separate ranking diagnostic rather than treating them as policy value.

## Reproduce the analysis

Python 3.11 through 3.13 is supported. LightGBM needs an OpenMP runtime; on macOS, `brew install libomp` provides it.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
python scripts/download_data.py
uplift-policy prepare --config configs/analysis.yaml
uplift-policy train --config configs/analysis.yaml
uplift-policy evaluate --config configs/analysis.yaml
pytest -q
```

The raw download, prepared data, and fitted models stay outside Git because of their size. The repository contains the aggregate tables and figures used in the paper.

## Dashboard

```bash
streamlit run app.py
```

The dashboard reads the saved aggregate results. It does not load user-level data or recalculate the analysis.

## Repository contents

| Path | Contents |
|---|---|
| `src/uplift_policy/` | Data preparation, diagnostics, models, estimators, bootstrap, and pipeline |
| `configs/analysis.yaml` | Seeds, split definition, model settings, capacities, and bootstrap settings |
| `results/tables/` | Aggregate estimates used in the paper and dashboard |
| `results/figures/` | Main result plots |
| `results/model_metadata.json` | Selected boosting rounds and validation losses |
| `paper/main.pdf` | Full paper by Yi Zhao |
| `tests/` | Tests for data rules, estimators, models, pipeline order, and dashboard loading |
| `data/README.md` | Official source, checksum, schema, citation, and dataset license |

## Data and license

The dataset is not copied into this repository. The download script obtains it from Criteo, and [data/README.md](data/README.md) records the exact artifact used. Criteo distributes the dataset under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The project code is available under the [MIT License](LICENSE).
