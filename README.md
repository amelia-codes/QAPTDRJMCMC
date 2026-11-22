# Quovar

Quovar builds a coherent random-walk step whose covariance follows the walk's
history, and studies how adaptive register grids (AMR) affect its accuracy.
The current results use two coordinates, three preparatory positions, and
8-bit position registers. Figures and insertable LaTeX tables are in
[`figures/quovar/`](figures/quovar/README.md) and [`paper/figures/`](paper/figures/).

## Run the results

Use Python 3.12 and the packages in `requirements.txt`. Figure rendering also
requires LaTeX with Computer Modern fonts. Run commands from the repository root.

```bash
# Redraw the ten figures and two tables from saved data.
python3.12 generate_paper_figures.py --plot-only

# Recompute every current experiment, then draw its figures and tables.
python3.12 generate_paper_figures.py

# Check the circuits, probability laws, and compressed history representation.
python3.12 -m unittest discover
```

The same seeds, measurement budgets, and mean ± one sample standard deviation
(`ddof=1`) are used when recomputing results. Each plot writes directly to
`figures/quovar/`; figure names are stable.

## Read the code in this order

[`generate_paper_figures.py`](generate_paper_figures.py) lists the experiments
in execution order. The implementation uses ordinary functions and arrays.

| File | What to read for |
| --- | --- |
| [`quantum_walk_study.py`](quantum_walk_study.py) | The physical problem, quantum circuit, decoded histories, covariance identity, and Gaussian-convergence experiment, in that order. Start with `make_walk`. |
| [`qcov.py`](qcov.py) | Integer addition, three-outcome quantum coins, and exact sampling of the constructed walk circuit. |
| [`covariance_benchmarks.py`](covariance_benchmarks.py) | The ten method labels, finite-shot measurement laws, and explicit comparison circuits. |
| [`qclt_proposal_comparison.py`](qclt_proposal_comparison.py) | Score the ten next-step methods against the same physical covariance, then write the table and bar chart. |
| [`qclt_step_comparison.py`](qclt_step_comparison.py) | Repeat the five readout methods at different measurement/oracle budgets. |
| [`amr_battle_test.py`](amr_battle_test.py) | Define the seven walk geometries, choose an AMR window, check coverage, and compare fixed and adaptive grids. |
| [`amr_tuning.py`](amr_tuning.py) | Compare three window widths and three choices of grid axes using matched history samples. |
| [`amr_criteria.py`](amr_criteria.py) | Decide whether a changed distribution needs refinement or coarsening. |
| [`paper_style.py`](paper_style.py), [`paper_costs.py`](paper_costs.py) | Shared plot appearance and circuit-stage resource counts. |

Circuit checks are collected in `test_quantum_walk_study.py` and
`test_covariance_benchmarks.py`. The explicit-history reference circuit lives
in the tests, alongside the compressed circuit it checks.

## The walk, mathematically

For a fixed history of positions $h_1,\ldots,h_K$, each history pair contributes
$r_{ab}(h_a-h_b)$, where $r_{ab}$ is $-1$, $0$, or $+1$ with probabilities
$q/2$, $1-q$, and $q/2$. The circuit sums these independent contributions.
Since

$$
\sum_{a<b}(h_a-h_b)(h_a-h_b)^T=K(K-1)S_h,
$$

choosing $q=\alpha/[K(K-1)]$ gives step covariance $\alpha S_h$.
The code adds a small isotropic lattice regularizer. This identity explains
the covariance scores without requiring a full statevector of the walk.
The circuit itself prepares coins and performs controlled integer additions.

The preparatory signs determine all three history positions, so the circuit
reconstructs pair differences from six coherent sign qubits. The final walk
uses 33 qubits with 8-bit coordinates. Independent repeated steps use fresh
coins. The convergence plot whitens the sum by its history-conditional
covariance before comparing it with a standard Gaussian.

AMR chooses the center, principal axes, and spacing from measurements of an
earlier circuit. That coordinate map is held fixed during the next coherent
walk. The [figure guide](figures/quovar/README.md) gives the comparison inputs,
readout costs, and definitions of every plotted error.

## Results and paper files

The entry point reads and writes six datasets in `data/quantum_walk/`:

| Dataset | Result |
| --- | --- |
| `results.npz` | Gaussian convergence of repeated steps |
| `covariance_errors.npz` | Ten-method comparison |
| `step_estimator_comparison.npz` | Measurement-budget comparison |
| `amr_battle_test.npz` | Fixed grids versus AMR across seven geometries |
| `amr_tuning.npz` | Window widths and grid axes |
| `amr_criteria.npz` | Refinement and coarsening thresholds |

The data files retain individual errors so plots calculate their means and
standard deviations directly. Older datasets and exploratory notebooks are
research records; the current entry point uses the six files above.

[`paper/`](paper/README.md) contains the manuscript copy, bibliography, tables,
and Quantikz/TikZ sources. The [literature review](docs/QCLT_LITERATURE_REVIEW.md)
maps claims to references. The [qubit audit](docs/QUBIT_AUDIT.md) and
[historical implementation audit](docs/README_FOR_CLAUDE.md) record the earlier
framework and its resource assumptions. Older application scripts were
removed from the active code; their committed versions remain in Git history.
