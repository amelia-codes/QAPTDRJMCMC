"""Recompute every result, then draw every figure and table into figures/.

    python3.12 generate_paper_figures.py              # recompute (about 2 minutes) and plot
    python3.12 generate_paper_figures.py --plot-only  # redraw from the saved data/
"""

import sys

import amr
import circuit_costs
import clt_convergence
import covariance_methods

for experiment in (covariance_methods, clt_convergence, amr):
    if "--plot-only" not in sys.argv:
        experiment.run()
    experiment.plot()
circuit_costs.write_table()
