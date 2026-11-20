"""Regenerate the current Quovar plots from saved experiment results."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_FIGURES = ROOT / "figures" / "quovar"
PLOT_OUTPUT = ROOT / "figures" / "quantum_walk"
PLOT_FILES = (
    "qclt_vs_covariance_methods.pdf",
    "qclt_step_generation.pdf",
    "qclt_amr_covariance_error.pdf",
    "qclt_amr_pilot_success.pdf",
    "qclt_amr_tilted_geometry.pdf",
    "qclt_clt_convergence.pdf",
    "amr_zoom_limit.pdf",
    "amr_methods.pdf",
    "amr_refinement_criterion.pdf",
    "amr_unrefinement_criterion.pdf",
)
SUPERSEDED_FILES = (
    "qclt_proposal_efficiency_d2.pdf",
    "qclt_proposal_efficiency_d5.pdf",
    "amr_regrid_interval.pdf",
    "amr_edge_threshold.pdf",
    "amr_pan_fraction.pdf",
    "amr_regrid_interval_narrow.pdf",
    "amr_regrid_interval_decoy.pdf",
    "amr_edge_threshold_narrow.pdf",
    "amr_edge_threshold_decoy.pdf",
    "amr_zoom_pan_narrow.pdf",
    "amr_zoom_pan_decoy.pdf",
)


def generate() -> list[Path]:
    """Render the insertable plots and remove their superseded versions."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/quovar-matplotlib")
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    PLOT_OUTPUT.mkdir(parents=True, exist_ok=True)

    import qclt_step_comparison
    import amr_battle_test
    import amr_tuning
    import amr_criteria
    import quantum_walk_study
    import qclt_proposal_comparison

    qclt_proposal_comparison.plot()
    qclt_step_comparison.plot()
    amr_battle_test.plot()
    amr_tuning.plot()
    amr_criteria.plot()
    quantum_walk_study.plot_clt_convergence()

    copied = []
    for name in PLOT_FILES:
        source = PLOT_OUTPUT / name
        if not source.is_file():
            raise FileNotFoundError(f"Plot {name} was not generated at {source}")
        target = PAPER_FIGURES / name
        shutil.move(source, target)
        copied.append(target)

    for name in SUPERSEDED_FILES:
        (PLOT_OUTPUT / name).unlink(missing_ok=True)
        (PAPER_FIGURES / name).unlink(missing_ok=True)
    return copied


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        raise SystemExit("Usage: python3.12 generate_paper_figures.py --plot-only")
    generated = generate()
    print(f"Generated {len(generated)} plots in {PAPER_FIGURES}")
