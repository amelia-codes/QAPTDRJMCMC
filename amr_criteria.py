"""Compare refinement/coarsening thresholds across QCLT walk distributions.

For each geometry, an existing 16-SD window faces either a narrowing or a
broadening history distribution. Three relative-width-change thresholds
decide whether to rebuild it. Every threshold sees identical old and new
measurement samples and is subject to the same circuit coverage guard.
"""

from pathlib import Path
import sys

import numpy as np

import amr_battle_test as study
from amr_tuning import _grouped_errors
import paper_style

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "quantum_walk" / "amr_criteria.npz"
THRESHOLDS = np.array([.10, .25, .50])
REPS = 80
HISTORY_SAMPLES = 1000
SCENARIOS = ("Narrowing", "Broadening")


def choose_window(current, proposed, threshold, direction, modes, histories):
    """Apply a width-change threshold, then the circuit's coverage guard.

    Widths are compared along the ordered principal axes. Refinement is
    triggered if a requested width shrinks by more than threshold; coarsening
    if one grows by more than threshold. Rebuilds use the measured center,
    principal axes, and 16-SD widths, exactly as in the original AMR study.
    """
    old_spacing = current[2]
    ratio = proposed[2] / old_spacing
    if direction == "Narrowing":
        triggered = bool(np.any(ratio < 1 - threshold))
    elif direction == "Broadening":
        triggered = bool(np.any(ratio > 1 + threshold))
    else:
        raise ValueError(f"Unknown width-change direction: {direction}")
    center, axes, spacing = proposed if triggered else current
    spacing = spacing.copy()
    guard_doublings = 0
    # The preparatory modes and their possible histories are specified by
    # this circuit. Coverage and the no-half-period-crossing condition are
    # checked independently of the scored covariance or any measured error.
    while not study.window_safe(modes, center, axes, spacing, histories):
        spacing *= 2
        guard_doublings += 1
        if guard_doublings > 16:
            raise ValueError("Could not construct a valid AMR window")
    return (center, axes, spacing), triggered, guard_doublings


def run():
    from quantum_walk_study import sample_walk

    rng = np.random.default_rng(20261122)
    scales = np.stack((rng.uniform(.45, .95, REPS), rng.uniform(1.05, 1.85, REPS)))
    shape = (len(study.CASES), len(SCENARIOS), len(THRESHOLDS), REPS)
    errors = np.empty(shape)
    updates = np.empty(shape, dtype=bool)
    guard_doublings = np.empty(shape, dtype=int)
    old_spacing = np.empty((len(study.CASES), len(SCENARIOS), REPS, study.D))
    proposed_spacing = np.empty_like(old_spacing)
    final_center = np.empty(shape + (study.D,))
    final_axes = np.empty(shape + (study.D, study.D))
    final_spacing = np.empty_like(final_center)
    heldout_mismatch = np.empty((len(study.CASES), len(SCENARIOS), 2))
    for case_index, (name, modes, pilot_width) in enumerate(study.CASES):
        pilot = sample_walk(
            study.PILOT_BITS, study.X0, np.eye(study.D),
            np.full(study.D, pilot_width / (1 << study.PILOT_BITS)),
            8, study.PILOT_POOL, 300 + case_index, modes_physical=modes)
        pool = pilot["physical_hist"]
        for scenario_index, direction in enumerate(SCENARIOS):
            check_frames = {}
            for replicate, scale in enumerate(scales[scenario_index]):
                before = pool[rng.choice(len(pool), HISTORY_SAMPLES, replace=False)]
                after = pool[rng.choice(len(pool), HISTORY_SAMPLES, replace=False)]
                # Scaling the physical coordinate map and preparatory modes
                # gives the exact history-measurement law of the rescaled
                # circuit. Each history sample retains its finite pilot-grid
                # rounding. The two sample sets are independent.
                after = study.X0 + scale * (after - study.X0)
                current = study.window_from_pilot(before)
                proposed = study.window_from_pilot(after)
                old_spacing[case_index, scenario_index, replicate] = current[2]
                proposed_spacing[case_index, scenario_index, replicate] = proposed[2]
                new_modes = scale * modes
                histories = study.all_true_histories(new_modes)
                for threshold_index, threshold in enumerate(THRESHOLDS):
                    frame, update, guarded = choose_window(
                        current, proposed, threshold, direction, new_modes, histories)
                    index = (case_index, scenario_index, threshold_index, replicate)
                    errors[index] = study.relative_error(new_modes, *frame)
                    updates[index] = update
                    guard_doublings[index] = guarded
                    final_center[index], final_axes[index], final_spacing[index] = frame
                    if replicate == 0 and threshold_index in (0, len(THRESHOLDS) - 1):
                        check_frames[threshold_index] = frame

            # Check both extreme criteria with a fresh coherent circuit.
            check_modes = scales[scenario_index, 0] * modes
            for j, threshold_index in enumerate((0, len(THRESHOLDS) - 1)):
                frame = check_frames[threshold_index]
                measured = sample_walk(
                    study.BITS, *frame, 14, 30_000,
                    1500 + 4 * case_index + 2 * scenario_index + j,
                    modes_physical=check_modes)
                if measured["operational_qubits"] != 33:
                    raise AssertionError("Changing the criterion changed the register width")
                heldout_mismatch[case_index, scenario_index, j] = measured["mismatch"]

        print(f"{name}: narrowing mean errors "
              + ", ".join(f"{value:.2%}" for value in errors[case_index, 0].mean(axis=1))
              + "; broadening "
              + ", ".join(f"{value:.2%}" for value in errors[case_index, 1].mean(axis=1)),
              flush=True)

    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA, names=np.array([case[0] for case in study.CASES]),
        scenarios=np.array(SCENARIOS), thresholds=THRESHOLDS, scales=scales,
        errors=errors, updates=updates, guard_doublings=guard_doublings,
        old_spacing=old_spacing, proposed_spacing=proposed_spacing,
        final_center=final_center, final_axes=final_axes, final_spacing=final_spacing,
        heldout_mismatch=heldout_mismatch, history_samples=HISTORY_SAMPLES,
        reps=REPS, register_bits=study.BITS, qubits=33,
        target_width_in_sd=16, seed=20261122)
    plot()


def plot():
    paper_style.apply()
    with np.load(DATA, allow_pickle=False) as saved:
        names = saved["names"].tolist()
        labels = [rf"Width-change threshold $= {100 * threshold:g}\%$"
                  for threshold in saved["thresholds"]]
        _grouped_errors(names, saved["errors"][:, 0], labels,
                        "amr_refinement_criterion.pdf")
        _grouped_errors(names, saved["errors"][:, 1], labels,
                        "amr_unrefinement_criterion.pdf")


if __name__ == "__main__":
    plot() if "--plot-only" in sys.argv else run()
