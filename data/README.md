# Dataset

This project uses the corrected `CRITEO-UPLIFTv2.1` dataset released by Criteo AI Lab. The data are downloaded from Criteo and are not stored in this repository.

## File used

- Dataset page: <https://ailab.criteo.com/criteo-uplift-prediction-dataset/>
- Download URL: <https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz>
- File name: `criteo-uplift-v2.1.csv.gz`
- File size: 311,422,618 bytes
- Rows: 13,979,592
- SHA-256: `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`

Download and verify the file with:

```bash
python scripts/download_data.py
```

The script checks the byte size, SHA-256 checksum, gzip stream, ordered schema, and row count. It saves the verified file under `data/raw/`, which Git ignores.

## Columns

The ordered columns are `f0` through `f11`, followed by `treatment`, `conversion`, `visit`, and `exposure`.

Criteo describes `f0`, `f2`, `f7`, and `f10` as continuous and the other feature columns as categorical. `treatment` is the randomized assignment used for the intention-to-treat analysis. `conversion` is the primary outcome and `visit` is secondary. `exposure` is measured after assignment, so it is excluded from every model and policy score.

## Citation and license

The dataset accompanies:

Eustache Diemert, Artem Betlei, Christophe Renaudin, and Massih-Reza Amini. "A Large Scale Benchmark for Uplift Modeling." AdKDD and TargetAd, 2018.

The corrected release is documented in:

Eustache Diemert, Artem Betlei, Christophe Renaudin, Massih-Reza Amini, Theophane Gregoir, and Thibaud Rahier. "A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift Modeling." 2021.

Criteo distributes the dataset under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
