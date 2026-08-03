PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all assets pdf arxiv clean

all: pdf

assets:
	cd .. && $(PYTHON) scripts/build_paper_assets.py

pdf: assets
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

arxiv: pdf
	COPYFILE_DISABLE=1 tar -czf arxiv-source.tar.gz \
		main.tex references.bib main.bbl figures tables

clean:
	$(LATEXMK) -c main.tex
