# Manuscript

`main.tex` is a standard single-column article that compiles with pdfLaTeX. It uses a fixed publication date, includes English and Chinese abstracts, and keeps all figures and generated tables inside this directory.

## Build

From the repository root:

```bash
python scripts/build_paper_assets.py
make -C paper pdf
```

The asset builder reads only the canonical aggregate files under `results/`. It creates the LaTeX tables, copies the required figures, and records their hashes in `assets_manifest.json`.

## arXiv source archive

```bash
make -C paper arxiv
```

This creates `paper/arxiv-source.tar.gz` with `main.tex`, the BibTeX database, the compiled bibliography, figures, and tables. It omits the compiled manuscript and local build files. Extracting the archive places `main.tex` at its root, which matches arXiv's compilation convention.

The manuscript has not been submitted to or endorsed by arXiv. The archive is a submission-ready source bundle that still requires the author's final account-level submission checks.

The manuscript source and compiled PDF are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
