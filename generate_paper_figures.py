"""Run the Quovar studies in reading order, or redraw their saved results.

    python3.12 generate_paper_figures.py              # recompute and plot
    python3.12 generate_paper_figures.py --plot-only  # use saved arrays
"""

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quovar-matplotlib")

import quantum_walk_study
import amr_battle_test
import amr_tuning
import amr_criteria
import qclt_proposal_comparison
import qclt_step_comparison
import paper_costs

PAPER_FIGURES = Path(__file__).parent / "figures" / "quovar"
PLOT_FILES = (
    "qclt_vs_covariance_methods.pdf", "qclt_step_generation.pdf",
    "qclt_amr_covariance_error.pdf", "qclt_amr_pilot_success.pdf",
    "qclt_amr_tilted_geometry.pdf", "qclt_clt_convergence.pdf",
    "amr_zoom_limit.pdf", "amr_methods.pdf",
    "amr_refinement_criterion.pdf", "amr_unrefinement_criterion.pdf",
)


def generate(plot_only=True):
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    if plot_only:
        quantum_walk_study.plot_clt_convergence()
        amr_battle_test.plot()
        amr_tuning.plot()
        amr_criteria.plot()
        qclt_proposal_comparison.plot()
        qclt_step_comparison.plot()
    else:
        quantum_walk_study.generate()
        amr_battle_test.run()
        amr_tuning.run()
        amr_criteria.run()
        qclt_proposal_comparison.generate()
        qclt_step_comparison.generate()
    paper_costs.generate()
    return [PAPER_FIGURES / name for name in PLOT_FILES]


if __name__ == "__main__":
    figures = generate(plot_only="--plot-only" in sys.argv)
    print(f"Generated {len(figures)} plots in {PAPER_FIGURES}")
