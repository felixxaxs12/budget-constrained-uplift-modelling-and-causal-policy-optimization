# Minimal project architecture

## Planned files

```text
configs/analysis.yaml
src/uplift_policy/data.py
src/uplift_policy/audit.py
src/uplift_policy/effects.py
src/uplift_policy/models.py
src/uplift_policy/evaluate.py
src/uplift_policy/pipeline.py
scripts/download_data.py
tests/
app.py
paper/
results/{tables,figures,manifests}/
```

The modules correspond to distinct responsibilities: data contracts, randomized-experiment audit, effect estimation, the locked model set, policy evaluation, and orchestration. They will be merged if implementation shows that a boundary contains only trivial forwarding code.

## Deliberately excluded

- Docker, DVC, MLflow, Airflow, dbt, Hydra, and a custom plugin system
- distributed computing or cloud infrastructure
- duplicate schema/config layers
- automatic recovery that changes data, models, or estimands
- speculative model families and ensembles
- a monetary optimizer without cost and value data

DuckDB handles ingestion and aggregate queries. Python, NumPy, scikit-learn, and LightGBM handle estimation. Streamlit is used only for the final read-only results explorer.
