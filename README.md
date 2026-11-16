# Quovar research repository

This repository contains the Quovar QCLT and adaptive register-grid study, together with earlier MCMC and gravitational-wave experiments.

## Layout

- [`paper/`](paper/): complete RevTeX manuscript, BibTeX references, and LaTeX source for the circuit and resource table.
- [`figures/quovar/`](figures/quovar/): current, paper-ready figures.
- [`figures/previous/`](figures/previous/README.md) and [`figures/quantum_walk_archive/`](figures/quantum_walk_archive/README.md): earlier results retained for reference.
- [`notebooks/exploratory/`](notebooks/exploratory/): research notebooks, grouped separately from source modules.
- [`docs/`](docs/): literature review, qubit audit, and notes on earlier implementation issues.
- [`data/`](data/): saved experiment outputs used by plot-only commands.
- Python source and experiment scripts remain in the repository root because they import one another directly.

## Build the paper

From the repository root:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The manuscript uses the `.bib` file in `paper/` and the PDFs in `figures/quovar/`.

## Redraw current figures

Run these commands from the repository root. They use saved experiment data and write PDFs to `figures/quantum_walk/`; copy the manuscript figures to `figures/quovar/` after regeneration.

```bash
python qclt_step_comparison.py --plot-only
python qclt_proposal_comparison.py --plot-only
python amr_battle_test.py --plot-only
python amr_tuning.py --plot-only
```

Plot labels use LaTeX through Matplotlib. Running new circuit simulations requires the project’s Qiskit environment; plot-only mode does not.
