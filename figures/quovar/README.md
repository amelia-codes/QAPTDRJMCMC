# Quovar paper figures

This directory contains the individual PDFs referenced by [`paper/main.tex`](../../paper/main.tex). The figures use LaTeX-rendered fonts, larger labels, Matplotlib's default color cycle, and green for AMR.

| Figure | File | Source |
| --- | --- | --- |
| QCLT circuit | `qclt_amr_circuit.pdf` | `figures/quantum_walk/qclt_amr_circuit.tex` |
| Covariance error versus measurement budget | `qclt_step_generation.pdf` | `qclt_step_comparison.py` |
| Covariance methods | `qclt_vs_covariance_methods.pdf` | `qclt_proposal_comparison.py` |
| AMR error across geometries | `qclt_amr_covariance_error.pdf` | `amr_battle_test.py` |
| AMR sample-count comparison | `qclt_amr_pilot_success.pdf` | `amr_battle_test.py` |
| Tilted covariance geometry | `qclt_amr_tilted_geometry.pdf` | `amr_battle_test.py` |
| Re-grid interval | `amr_regrid_interval_{narrow,decoy}.pdf` | `amr_tuning.py` |
| Edge criterion | `amr_edge_threshold_{narrow,decoy}.pdf` | `amr_tuning.py` |
| Window width and pan fraction | `amr_zoom_pan_{narrow,decoy}.pdf` | `amr_tuning.py` |

The plot-only commands use saved numerical results:

```bash
python qclt_step_comparison.py --plot-only
python qclt_proposal_comparison.py --plot-only
python amr_battle_test.py --plot-only
python amr_tuning.py --plot-only
```

The first three commands write their PDFs to `figures/quantum_walk/`; the AMR refinement plots are written there as well. Copy selected manuscript PDFs into this directory after regeneration. `qclt_step_comparison.py` and `qclt_proposal_comparison.py` need Qiskit only when running new circuit experiments, not when plotting saved data.
