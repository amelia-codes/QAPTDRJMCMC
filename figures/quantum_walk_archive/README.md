# Coherent QCLT covariance walk and AMR

**Archive.** These diagnostic plots preceded the current standalone plots in
[`figures/quantum_walk/`](../quantum_walk/README.md). The five-case AMR stress
plot is retained for provenance; the current `amr_battle_test.py` regenerates
the seven-case tradeoff plot in the primary directory.

Older combined-panel PDFs that were replaced by standalone plots are stored in
[`legacy_composites/`](legacy_composites/).

The separate full-history efficiency PDFs were folded into the stored-history
proposal plots at $K=100$ and retained in
[`legacy_full_history/`](legacy_full_history/) for reference.

These are focused, two-dimensional demonstrations of a QCLT walk. There is no
gravitational-wave model, Metropolis acceptance, likelihood, tempering, or
reversible-jump step. The circuits are built in
[`quantum_walk_study.py`](../../quantum_walk_study.py). Each circuit starts from
a basis position, makes three coherent preparatory random-walk steps, and then
makes one or more coherent covariance-shaped steps. It contains no measurement
or reset before its final readout.

The random-walk histories are determined by six quantum preparatory coin
qubits: two mode signs for each of three steps. The pair differences needed by
QCLT are calculated coherently from those coins. This removes the need to
store three separate position snapshots. A one-qubit parity scratch register
is uncomputed after each term. This compression is exact for the specified
walk whose initial position and step modes are known. It is not a way to store
arbitrary independent, unknown quantum history states in six qubits.

For history positions \(h_1,h_2,h_3\), the adaptive kick is

\[
\epsilon=\sum_{a<b} r_{ab}(h_a-h_b)+\eta,
\qquad
\mathrm{Cov}(\epsilon\mid h)=\frac{2.38^2}{2}\,S_h+0.1I
\]

in grid coordinates. Each \(r_{ab}\) is a fresh three-outcome quantum coin;
\(\eta\) uses independent coins for the small isotropic floor. The three
preparatory steps use two coupled physical displacement modes,
\((0.0020,0.0012)\) and \((0.0007,0.0015)\). The continuous-grid target
is their expected history covariance, scaled by \(2.38^2/2\). The small
isotropic floor is included in the actual circuit but not in that target,
so its physical size depends on grid spacing. The experiment measures how
well a grid represents the intended narrow physical covariance.

## Figures

- [One-page summary](qclt_amr_summary.pdf): the four main results together:
  history-conditioned next steps, the measured AMR window, error versus
  qubits, and repeated coherent steps. **The circuit outputs random steps,
  never a covariance matrix;** the plotted covariances are calculated
  afterward from final samples only to validate those steps.
- [Coherent QCLT covariance](qclt_coherent_covariance.pdf): the measured
  conditional second moments agree with the history-dependent circuit
  prediction for all 64 observed history branches. The right panel shows
  bootstrap sampling error, with 32 resamples per sample size and the
  10th–90th percentile range.
- [QCLT with AMR](qclt_amr_grid.pdf): final measurements of a **pilot
  circuit** determine the next circuit's window center, principal axes, and
  spacing. The pilot window is centered on the known initial point and has
  width 0.08 on each physical axis. A full-range 6-bit grid rounds both
  narrow preparatory modes to zero; the pilot window resolves them, and
  measured AMR improves the physical covariance further. Finer 8-bit and
  12-bit grids use the **same physical AMR window** and recover more
  accuracy. The 16-bit full-range circuit is an accuracy reference.
- [Repeated coherent kicks](qclt_multistep_walk.pdf): one 8-bit AMR circuit
  contains one to four adaptive kicks with fresh coins. Their preparatory
  history is fixed for this segment. The sampled total variance and cross
  covariance grow as the conditional covariance predicts. AMR selection
  still occurs only between circuits.
- [AMR stress test](qclt_amr_battle_test.pdf): five covariance geometries,
  seven pilot budgets, and 80 independently selected pilot subsets per
  setting. Untuned measured AMR is compared with both a full-range grid and
  an informed axis-aligned local grid at the same 33-qubit final width.
  The right panel counts unsafe windows as unsuccessful trials.
- [Correlated-Gaussian output-law replica](qclt_vs_gaussian_shear.pdf):
  a Kitaev–Webb-style independent-Gaussian preparation and reversible shear
  on the same small local grids. Its covariance is supplied classically and
  its loader uses generic state preparation; it is a comparator for output
  law, not a complete history-adaptive quantum walk. See the
  [literature and matched-test review](../../docs/QCLT_LITERATURE_REVIEW.md).

The relative errors below compare each circuit's **exact unconditional
physical step covariance** with the continuous-grid target. This covariance
can be calculated analytically from the preparatory coin distribution, so
the tabulated error has no sampling noise. Separately, 100,000 exact samples
of each final measurement distribution verify that the observed covariance
matches the circuit prediction. The diagnostic displacement tracker is added
solely for reading out an unwrapped step and has been checked not to change
the position output.

| Grid and window | Operational qubits | With diagnostic tracker | Relative covariance error |
| --- | ---: | ---: | ---: |
| Full-range 6-bit | 29 | 45 | 1.770 |
| Pilot 6-bit window | 29 | 45 | 0.509 |
| Measured AMR 6-bit window | 29 | 45 | 0.092 |
| Same AMR window, 8-bit | 33 | 61 | 0.010 |
| Same AMR window, 12-bit | 41 | 77 | 0.0017 |
| Full-range 16-bit | 49 | 89 | 0.0017 |

This table uses **one** 100,000-shot pilot window. In the independent-pilot
stress test, the tilted 8-bit AMR median is 2.61% across 80 pilot subsets of
1,000 samples, with a 10th–90th percentile range of 1.14%–6.03%. The 0.955%
entry here is favorable for that one pilot and should not be presented as a
typical outcome. The stress test uses the original measured-covariance
window rule, with no secondary grid tuning.

The repeated-kick 8-bit AMR circuits use 33, 43, 53, and 63 operational
qubits for one through four kicks, respectively. They remain under 100
logical qubits, including the diagnostic tracker (61 to 91 qubits).

## Qubit budget and scope

| Purpose, one 6-bit QCLT+AMR kick | Qubits |
| --- | ---: |
| Current two-coordinate position | 12 |
| Six preparatory signs encoding all three histories | 6 |
| Three QCLT pair move coins and three pair sign coins | 6 |
| Two isotropic move coins and two isotropic sign coins | 4 |
| Reusable parity scratch | 1 |
| **Operational circuit** | **29** |
| Optional two 8-bit displacement readout registers | 16 |
| **Diagnostic circuit** | **45** |

Each additional adaptive kick needs ten fresh coins: six for the three
history pairs and four for the isotropic term. The 16-bit reference replaces
the 12 position qubits with 32, for 49 operational qubits. The logical counts
are Qiskit circuit widths. The controlled arithmetic can be deep when
decomposed to hardware gates; this study does not report physical hardware
requirements or noise tolerance. For example, the one-kick 6-bit AMR circuit
has 534 high-level gates and depth 362 before decomposition; the four-kick
8-bit AMR circuit has 3,006 gates and depth 2,113. The full-range 16-bit reference has 4,140
high-level gates and depth 2,907. These depths include multi-controlled X
gates as single operations, so hardware-native depths will be larger.

AMR is **classical between coherent circuit executions**, as requested: the
pilot's final measurements are processed with NumPy to set the next window.
Within each circuit, the adaptive step is history-dependent and coherent.
These circuits have no interfering computational branches, so
[`qcov.sample_circuit`](../../qcov.py) draws exactly from their ideal final
measurement distribution without a statevector or quantum device. These
figures are circuit simulations and do not establish a quantum speedup.

To reproduce from the repository root in an environment with Qiskit, NumPy,
and Matplotlib:

```bash
python quantum_walk_study.py
```

The summary arrays and exact sample statistics are in
[`data/quantum_walk/results.npz`](../../data/quantum_walk/results.npz).
`python quantum_walk_study.py --plot-only` redraws the PDFs from that file.
[`test_quantum_walk_study.py`](../../test_quantum_walk_study.py) compares the
compressed circuit with an independent circuit that explicitly stores every
history snapshot and checks exact output equality for the same coin outcomes.
It can be run with `python test_quantum_walk_study.py`.

Run `python amr_battle_test.py` for the pilot stress test and
`python gaussian_shear_replica.py` for the Gaussian output-law replica. Their
summary arrays are in [`data/quantum_walk/`](../../data/quantum_walk/).
