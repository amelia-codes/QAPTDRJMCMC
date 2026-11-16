# Paper files

- `main.tex` is the complete RevTeX manuscript.
- `references.bib` is the BibTeX database.
- `figures/qclt_amr_circuit.tex` is the standalone Quantikz circuit source.
- `figures/method_costs.tex` contains the resource-cost table source.
- `figures/covariance_method_errors.tex` contains the full covariance-error table, generated from saved comparison data.

The manuscript refers to figure PDFs in `../figures/quovar/`. Build from this directory with:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Regenerate the covariance-error table from the saved comparison data from the repository root with `python qclt_proposal_comparison.py --table-only`.
