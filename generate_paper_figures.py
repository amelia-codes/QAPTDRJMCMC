# Recompute every result, then draw every figure, diagram and table into figures/
#
#   python3.12 generate_paper_figures.py              # recompute (about 7 minutes) and plot
#   python3.12 generate_paper_figures.py --plot-only  # redraw from the saved data/

import sys

import amr
import circuit_costs
import circuit_diagrams
import clt_convergence
import covariance_methods
import random_pairs


if __name__ == "__main__":
    for experiment in (covariance_methods, clt_convergence, random_pairs, amr):
        if "--plot-only" not in sys.argv:
            experiment.run()
        experiment.plot()

    circuit_costs.write_table()
    circuit_diagrams.plot()
