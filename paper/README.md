# Paper files

- `main.tex` is the complete RevTeX manuscript.
- `references.bib` is the BibTeX database.
- `figures/qclt_amr_circuit.tex` is the standalone Quantikz circuit source.
- `figures/method_costs.tex` contains the resource-cost table source.

The manuscript refers to figure PDFs in `../figures/quovar/`. Build from this directory with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
