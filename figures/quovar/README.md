# Quovar paper figures

This directory contains the individual PDFs referenced by [`paper/main.tex`](../../paper/main.tex). The figures use LaTeX-rendered fonts, larger labels, Matplotlib's default color cycle, and green for AMR.

| Figure | File | Source |
| --- | --- | --- |
| QCLT circuit | `qclt_amr_circuit.pdf` | `figures/quantum_walk/qclt_amr_circuit.tex` |
| Covariance error versus measurement budget | `qclt_step_generation.pdf` | `qclt_step_comparison.py` |
| Full covariance-method results | [`covariance_method_errors.tex`](../../paper/figures/covariance_method_errors.tex) | `qclt_proposal_comparison.py --table-only` |
| AMR error across geometries | `qclt_amr_covariance_error.pdf` | `amr_battle_test.py` |
| AMR sample-count comparison | `qclt_amr_pilot_success.pdf` | `amr_battle_test.py` |
| Tilted covariance geometry | `qclt_amr_tilted_geometry.pdf` | `amr_battle_test.py` |
| Re-grid interval | `amr_regrid_interval_{narrow,decoy}.pdf` | `amr_tuning.py` |
| Edge criterion | `amr_edge_threshold_{narrow,decoy}.pdf` | `amr_tuning.py` |
| Window width and pan fraction | `amr_zoom_pan_{narrow,decoy}.pdf` | `amr_tuning.py` |

The plot-only commands use saved numerical results:

```bash
python qclt_step_comparison.py --plot-only
python qclt_proposal_comparison.py --plot-only  # stored-history proposal-efficiency plots
python amr_battle_test.py --plot-only
python amr_tuning.py --plot-only
```

The plotting commands write PDFs to `figures/quantum_walk/`; copy selected manuscript PDFs into this directory after regeneration. The covariance-method comparison is included as a LaTeX table, not as a plot. `qclt_step_comparison.py` and `qclt_proposal_comparison.py` need Qiskit only when running new circuit experiments, not when plotting saved data.
