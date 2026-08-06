# Paper

The full paper is available as [main.pdf](main.pdf). Its LaTeX source is [main.tex](main.tex).

From the repository root, rebuild the tables and PDF with:

```bash
python scripts/build_paper_assets.py
make -C paper pdf
```

The table builder reads the aggregate CSV files in `results/tables/` and the selected model settings in `results/model_metadata.json`. The manuscript reads figures directly from `results/figures/`.
