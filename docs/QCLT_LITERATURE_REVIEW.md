# Covariance-shaped quantum proposals: literature and matched tests

## Question and comparison rule

The task here is to **produce the next random step** whose covariance follows
the recent walk history. Measuring a covariance matrix is a different task.
Accordingly, a method is an end-to-end comparator only if it accepts a
history (or an explicitly stated covariance oracle) and prepares or samples a
next-step distribution. We score the covariance of that final distribution
*afterward* as a diagnostic. The focused experiment has two coordinates,
three preparatory positions, and a small isotropic regularizer. It has no
Metropolis accept/reject operation or target density.

For history \(h_1,\ldots,h_K\), let \(S_h\) be the unbiased sample covariance.
The intended conditional step covariance on the encoded lattice is
\(C_h=(2.38^2/d)S_h+0.1I\), with \(d=2\), \(K=3\). The physical target used in
the [walk figures](../figures/quovar/README.md) is the expected
continuous-grid history term; it excludes the lattice regularizer. That
target is deliberately narrower than the physical range represented by a
full-range low-bit grid. The familiar \(2.38^2/d\) scale comes from
high-dimensional random-walk Metropolis theory under particular target
assumptions, not from a general quantum theorem
([Roberts, Gelman and Gilks, 1997](https://sites.stat.columbia.edu/gelman/research/published/theory7.pdf)).

## Main approaches

| Route | How it supplies the next step | Information supplied | Cost and accuracy issue |
| --- | --- | --- | --- |
| Classical adaptive Metropolis | Estimate \(S_h\), factor it, draw a Gaussian, optionally accept/reject | Classical history | \(O(Kd^2+d^3)\) for a direct covariance plus Cholesky update, then \(O(d^2)\) per Gaussian draw; an \(O(Kd)\) low-rank draw is possible when \(K\ll d\). Sampling error in \(S_h\) remains. This is the natural classical reference ([Haario, Saksman and Tamminen, 2001](https://researchportal.helsinki.fi/en/publications/an-adaptive-metropolis-algorithm/)). |
| Classical pair-difference sampler | Draw the same independent three-outcome pair coins and add their signed history differences | Classical history | \(O(K^2d)\) arithmetic per step with the unoptimized pair list, and exactly the same next-step law as the QCLT circuit when only its position is measured. It needs no explicit covariance matrix. This is the strongest classical control for any claimed speed advantage. |
| QCLT pair-difference step, this repository | Coherent coins choose signed pair differences and update a quantum position register | Coherently encoded history, or the six preparatory signs used here | Conditional covariance is exactly \(C_h\) on the rounded lattice. It does **not** make a Gaussian law, remove history sampling error, or by itself accelerate a classical calculation. Width is 29, 33, 41, and 49 logical qubits for 6, 8, 12, and 16 bits per coordinate with one kick. |
| QCLT + AMR, this repository | Same QCLT step, with grid center, axes and spacing chosen from final measurements of an earlier pilot circuit | Measured pilot histories between circuits | Same final-circuit width as QCLT at a fixed bit count. Smaller rounding error is possible, but pilot samples, eigendecomposition, window-safety checks and circuit recompilation cost time. The window update is classical. |
| Correlated Gaussian state preparation | Prepare independent one-dimensional Gaussian states and reversibly shear them into a correlated quantum proposal | Covariance/Cholesky-like parameters, usually classical | Direct next-step comparator, provided a covariance is already available. Finite-grid rounding and state-preparation error matter. Fully history-controlled preparation requires reversible covariance arithmetic and controlled loading not supplied for free. [Kitaev and Webb](https://arxiv.org/abs/0801.0342); [Bauer *et al.*](https://arxiv.org/abs/2109.10918). |
| Covariance density matrix and quantum PCA | Amplitude encode centered observations and prepare a density matrix proportional to covariance; diagonalize in a quantum representation | Efficient amplitude access to centered data, copies for PCA | Produces spectral information, **not** a sampled next step. A square-root transform and proposal loader would still be needed. Uncentered data require care ([Gordon *et al.*, 2022](https://arxiv.org/abs/2204.03495); [Lloyd, Mohseni and Rebentrost, 2013](https://arxiv.org/abs/1307.0401)). |
| Quantum expectation or amplitude estimation | Estimate covariance entries, then use a separate Gaussian proposal circuit | Reversible sample oracle and its inverse | Scalar bounded expectations can have \(O(1/\varepsilon)\), rather than \(O(1/\varepsilon^2)\), precision query scaling. Covariance needs up to \(d(d+1)/2\) entries and a proposal loader; oracle preparation, numeric precision, confidence and dimension affect total cost. The estimate is an intermediate classical output, outside the preferred next-step-only scope ([Brassard *et al.*](https://arxiv.org/abs/quant-ph/0005055); [Montanaro](https://arxiv.org/abs/1504.06987); [Cornelissen and Jerbi](https://arxiv.org/abs/2107.03410)). |

Multivariate mean estimation can improve precision query complexity under
particular oracle models, while dimension costs and lower bounds remain
important ([Cornelissen, Hamoudi and Jerbi](https://arxiv.org/abs/2111.09787)).
The many covariance-estimation variants in
[`cov_methods.py`](../cov_methods.py) therefore do **not** count as direct
next-step baselines here. They are possible components of a longer pipeline.

Quantum Metropolis-Hastings walks address a different layer: they quantize a
proposal and acceptance kernel. They require a suitable proposal oracle and
often substantial reversible arithmetic
([Lemieux *et al.*](https://arxiv.org/abs/1910.01659)). Spectral-gap speedups
for fixed reversible chains cannot simply be assigned to an adaptively
changing AMR walk ([Szegedy](https://arxiv.org/abs/quant-ph/0401053)).
Our current QCLT circuit leaves its coin/history registers entangled with
the position; it is a coherent proposal experiment, not yet a cleaned
Szegedy transition oracle.

## Published Gaussian preparation cost, specialized to our small case

[Bauer *et al.*](https://arxiv.org/pdf/2109.10918) give an explicit
Kitaev–Webb shearing construction for \(d\) coordinates of \(k\) qubits each.
Its shear uses \(r=(k-1)+\lceil\log_2(d-1)\rceil\) precision bits and at most
\((d^2-d)(4k^2+8r^2+8kr+26r+11k-8)\) CNOTs, with \(2r+1\) ancillae beyond
the \(dk\) state qubits. This bound **excludes** one-dimensional Gaussian
preparation and any coherent calculation of covariance from history.

| \(d=2\) | State qubits | Shear ancillae | Total for prepared state plus shear | Published CNOT upper bound for shear only |
| --- | ---: | ---: | ---: | ---: |
| \(k=6\) | 12 | 11 | 23 | 1,544 |
| \(k=8\) | 16 | 15 | 31 | 2,716 |

These are meaningful width comparisons with a **precomputed** covariance;
they are not circuit-size claims for a coherent history-adaptive walk.
Bauer *et al.* also observe that generic short-register state loading can be
cheaper than the asymptotically efficient one-dimensional Gaussian loader at
small \(k\), while optimized shear arithmetic becomes useful as the number
of coordinates grows. QCLT's 33-qubit 8-bit circuit includes its history and
proposal coins, so its width is similarly modest but its arithmetic is deep.
For example, decomposing one local-grid 8-bit QCLT circuit with Qiskit 1.1.0's
`u,cx` basis and zero optimization gives 57,806 CX gates, 121,873 gates in
total, and depth 69,065. These are compilation diagnostics, not optimized
hardware estimates. The published Gaussian shear CNOT bound omits loading
and history processing, so comparing these two numbers as an end-to-end
speedup would be invalid.

## Matched numerical experiments in this repository

The [direct-step comparison](../figures/quovar/qclt_step_generation.pdf),
run by [`qclt_step_comparison.py`](../qclt_step_comparison.py), scores all 64
history branches against the **same unrounded physical conditional target**.
The ideal QCLT output law has 8.8% median physical covariance error at 8
bits per coordinate. It is exact only relative to its *rounded-lattice*
history covariance. At 384 intermediate shots, measured-entry pairwise and
SWAP covariance estimators followed by an **ideal downstream Gaussian
loader** have 11.8% and 9.3% median physical step-covariance errors. These
percentages measure the covariance of the resulting *step law*, not a
covariance matrix as the algorithm's output. The pairwise and SWAP ancilla
probabilities are verified with actual five-qubit statevector circuits;
the plotted shot noise is sampled analytically from those probabilities.
The measured-entry methods are given each history branch classically,
and SWAP is given exact variances, so this is an optimistic comparison for
them. Their curves converge toward the same grid-error floor as the shot
budget grows.

The figure now includes an explicit **non-equal qubit-width table**. The
QCLT circuit uses 33 logical qubits for its complete known-mode conditional
step. The pairwise and SWAP readout circuits each use 5 qubits, but only
produce covariance entries; they still need a Gaussian step loader (at
least 16 state qubits for two 8-bit coordinates). The published 31-qubit
Kitaev–Webb state-plus-shear stage presumes covariance supplied classically
and excludes one-dimensional loading ancillae. No complete coherent
history-conditioned estimator-plus-loader implementation was built or
costed. This figure therefore cannot establish that QCLT uses fewer qubits
or gives a general computational speedup. Exact classical sample covariance
and classical pair-difference sampling can also avoid the finite-shot
readout error.

The [AMR covariance-error comparison](../figures/quovar/qclt_amr_covariance_error.pdf) and
[AMR pilot-reliability comparison](../figures/quovar/qclt_amr_pilot_success.pdf), run by
[`amr_battle_test.py`](../amr_battle_test.py), holds the final QCLT circuit at
8 bits per coordinate and 33 logical qubits. It compares the full physical
range, an informed axis-aligned local window with the same pilot width, and
the measured principal-axis AMR window. Seven step geometries probe tilt,
sign, isotropy, scale, and two fixed-grid-friendly regimes. For each geometry, 80 independent subsets of
pilot final measurements are used at each of seven pilot budgets. Errors
are exact circuit-law covariance errors relative to the continuous target;
unsafe windows count as unsuccessful trials. One separate 30,000-shot
circuit-law check per geometry validates the analytic scoring.

| Geometry | Full-range 8-bit error | Informed local 8-bit error | Untuned AMR 8-bit median error, 1,000 pilots |
| --- | ---: | ---: | ---: |
| Tilted | 169.9% | 9.04% | 2.61% |
| Axis-aligned | 264.8% | 11.29% | 3.59% |
| Anti-tilted | 169.9% | 9.04% | 2.81% |
| Isotropic | 64.1% | 8.74% | 5.50% |
| Broader | 16.9% | 9.04% | 3.24% |
| Grid-aligned | 152.8% | 0.114% | 4.88% |
| Isotropic, already resolved | 157.1% | 0.083% | 5.12% |

At 1,000 pilots, the tilted AMR 10th–90th percentile is 1.14%–6.03%; the
one 100,000-shot pilot in the original figure happened to give 0.955%.
The measured rule beat the informed local grid in 79–80 of 80 runs for each
of the first five geometries at the largest pilot budget, but lost all 80
trials in each of the two well-resolved geometries. The result is thus
geometry-dependent. At tiny pilot budgets,
sample noise can shift the AMR axes or window; the plotted win-rate panel
includes unsafe windows as failures. The “population geometry” reference
uses the true history distribution and is a limiting *geometry*, not an
accuracy lower bound: integer rounding can make a finite-sample orientation
luckier than the limiting one.

An exploratory grid “tuning” pass has been removed. It optimized the final
window to reproduce the **6-bit pilot step covariance**. That pilot step
already has 50.9% error relative to the continuous physical target, so the
optimization rewarded preservation of coarse-grid bias. It repeatedly made
the final result worse. The plots and arrays linked here use only the
original pilot-history window rule; no output covariance is extracted as the
algorithm's answer.

The [Kitaev–Webb-style replica](../figures/quantum_walk_archive/qclt_vs_gaussian_shear.pdf),
run by [`gaussian_shear_replica.py`](../gaussian_shear_replica.py), prepares two
one-dimensional Gaussian amplitude states using generic `StatePreparation`,
then applies a reversible, lookup-controlled shear. It uses the **known
unconditional rounded-lattice covariance supplied classically** on the same
6-bit local grids. This reproduces the published independent-Gaussians-plus-
shear *output law*, not the paper's optimized arithmetic or a coherent
history-conditioned covariance loader. Its 12 state qubits and exact
statevector agree with an independent analytic convolution. Relative to the
ideal **rounded-lattice** covariance, the direct discretized Gaussian has
negligible error in these cases; the explicit shear has 5.83% covariance
error and 1.93% infidelity on each tilted case, and negligible error when
the axes are independent. On the tilted case, the analytic rounded-shear
covariance error falls to 0.76%, 0.27%, 0.035%, and 0.015% at 7–10 bits per
coordinate. Fidelity need not improve monotonically at every bit count
because the shear coefficient is rounded against a changing lattice.

The Gaussian replica is an optimistic *distribution-preparation* comparator:
it is handed a covariance that the QCLT circuit computes coherently from
its history. It also targets a Gaussian law whereas QCLT targets the same
second moment with a discrete pair-difference law. The replica's 5.83%
rounding error uses the **lattice** covariance denominator; the AMR table
uses a **continuous physical** denominator. The two percentages should not
be read as a head-to-head ranking. An end-to-end Gaussian comparison must
include reversible history covariance computation and covariance-controlled
state preparation, or measure/compute covariance classically and be labelled
hybrid.

The AMR gain itself is a **grid-resolution** effect and is not unique to
QCLT: a Gaussian step sampled from the same rounded history covariance on
the same physical window inherits essentially the same physical covariance
rounding error. QCLT's distinguishing feature is how it produces the
history-conditioned second moment without first materializing covariance
entries. Neither this stress test nor the Gaussian replica demonstrates a
speed advantage over classical adaptive sampling.

The explicit 6-bit lookup circuit is deliberately simple and expensive:
Qiskit 1.1.0 zero-optimization `u,cx` transpilation gives 748,520 CX gates,
1,499,445 gates in total, and depth 1,242,854 on the tilted case. This is
**not** Bauer *et al.*'s efficient shearing implementation.
It validates the output law and illustrates why high-level `StatePreparation`
or controlled additions cannot be counted as unit-cost hardware gates.

## Matched comparison with covariance-based random walks

[`qclt_proposal_comparison.py`](../qclt_proposal_comparison.py) compares QCLT + AMR with the proposals listed
below, at matched information:
- **Classical adaptive proposals:**
  - Adaptive Metropolis (Haario, Saksman & Tamminen, *Bernoulli* 7, 2001), with the Roberts & Rosenthal
    safety component (*J. Comput. Graph. Stat.* 18, 2009);
  - AM with Ledoit–Wolf shrinkage of the covariance (*J. Multivariate Anal.* 88, 2004);
  - the single pair-difference move of DE-MC / DE-MCz (ter Braak, *Stat. Comput.* 16, 2006; ter Braak & Vrugt,
    *Stat. Comput.* 18, 2008), which is QCLT's closest classical relative.
- **Naive approximations:** isotropic, diagonal-only, zero-mean (uncentred), and the max/min Cauchy–Schwarz
  bounds.
- **Hybrid readout pipelines:** centred pairwise moments, SWAP-test distance estimation, and canonical
  amplitude estimation (Brassard *et al.*), each followed by an ideal Gaussian loader.
- **A window-only kick on the AMR grid.**

Proposal quality is scored as jump distance along the worst-explored direction per likelihood evaluation, a
worst-direction variant of the expected squared jumped distance of Pasarica & Gelman (*Statistica Sinica* 20,
2010). Results are in [the figure guide](../figures/quovar/README.md), sections 3–4. In short:
- **Following a coherently held history** (no readout), QCLT + AMR gives the most accurate next-step
  covariance of all the step generators tested.
- **The lazy coin wastes most proposals in five dimensions.** A full-strength version (every pair active,
  output scaled) fixes this and then matches AM and the single-pair move built from the same K stored
  states.
- **When the history is classical,** a window-only kick on the AMR grid, which needs no history registers,
  matches full-history AM and outperforms QCLT. None of this is a quantum speedup.

## What the evidence supports

The clean finding is a resource/accuracy trade: a 33-logical-qubit coherent
history-dependent QCLT proposal can represent the intended covariance much
better when a measured pilot selects an appropriate physical grid. For five
controlled geometries, the improvement survives repeated pilot draws and
an informed local-grid baseline; for two well-resolved geometries AMR is
substantially worse. The quantum circuit supplies a step,
not an extracted covariance matrix. A practical advantage over a classical
adaptive Gaussian or pair-difference sampler, and a quantum mixing-time
speedup, remain unestablished. Classical sampling from the same small coin
law is straightforward; QCLT's value as a proposal subroutine would require
an application where coherent access to the step and subsequent interference
are actually used.

## Reproduce

With Python 3.12, Qiskit, NumPy, and Matplotlib installed:

```bash
python3.12 test_quantum_walk_study.py
python3.12 qclt_step_comparison.py
python3.12 amr_battle_test.py
python3.12 gaussian_shear_replica.py
```

Saved arrays are in [`data/quantum_walk/`](../data/quantum_walk/); the two primary
PDFs are in [`figures/quantum_walk/`](../figures/quantum_walk/) and earlier plots
are in the [adjacent archive](../figures/quantum_walk_archive/). The Gaussian statevector
check is in the replica script. The AMR test is deterministic given its
listed random seeds and pilot budgets.
