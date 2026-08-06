.PHONY: pdf tables clean

pdf: tables
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

tables:
	cd .. && python3 scripts/build_paper_assets.py

clean:
	latexmk -c main.tex
