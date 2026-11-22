# Covariance-shaped quantum proposals: literature and matched tests

The covariance-scaled coin sum ("coin sum" below) was called QCLT in earlier versions.

## Citation audit (21 November 2026)

The six papers originally in `paper/references.bib` (kept outside this repository)
are real and relevant to the claims attached to them in the local manuscript.
The main improvements are to correct author metadata and cite the methods
actually used in the comparisons. For the focused coin sum + AMR paper, adaptive
Metropolis, differential-evolution proposals, and quantum Gaussian preparation
are more central than the BayesWave and transdimensional motivation.
The recommendations below use the local manuscript and implementation;
the newer Overleaf manuscript was not available for this audit. Existing
citation keys are preserved.

### Corrections to the original bibliography

| Existing key | Finding and recommended use | Verified source |
| --- | --- | --- |
| `cornish2015bayeswave` | Keep for the original BayesWave algorithm and application motivation. The author, year, and DOI fields were correct. | [Cornish and Littenberg (2015)](https://arxiv.org/abs/1410.3835) |
| `becsy2017parameter` | Keep for BayesWave performance and parameter recovery if that discussion remains. Restored all nine authors in their published order, including Peter Raffai as second author; corrected Jonah Kanner's name. | [Bécsy et al. (2017)](https://arxiv.org/abs/1612.02003) |
| `haario2001adaptive` | Keep as the main source for learning Gaussian proposal covariance from chain history. Metadata were correct. | [Haario, Saksman and Tamminen (2001)](https://researchportal.helsinki.fi/en/publications/an-adaptive-metropolis-algorithm/) |
| `terbraak2008demcz` | Keep for proposals made from differences of stored past states. Add the original 2006 DE-MC paper alongside it. | [ter Braak and Vrugt (2008)](https://link.springer.com/article/10.1007/s11222-008-9104-9) |
| `tian2024adaptive` | Corrected all three given names to **Zhiyao Tian, Anthony Lee, and Shunhua Zhou**. Appropriate for adaptive tempered RJMCMC in curve fitting. Their covariance is learned on a grid of function values; that grid is a different construction from our moving quantum register window. | [Tian, Lee and Zhou (2024)](https://arxiv.org/abs/2402.08844) |
| `lemieux2020efficient` | Corrected **Jessica Lemieux** and **Bettina Heim**. Keep for explicit quantum Metropolis–Hastings walk circuits and their arithmetic costs. | [Lemieux et al. (2020)](https://quantum-journal.org/papers/q-2020-06-29-287/) |

### Sources to use for the coin sum and AMR comparisons

These entries have been added to the BibTeX database. Select citations by the
claim being made; adding an entry to the database does not add it to the paper.

| Claim or method | Recommended citation keys | Why this is the right source |
| --- | --- | --- |
| History-difference proposals | `terbraak2006demc,terbraak2008demcz` | The [original DE-MC paper](https://link.springer.com/article/10.1007/s11222-006-8769-1) introduces population-difference proposals; the 2008 extension uses stored past states. These are the closest classical precedents for our step construction. Our sum of pair contributions should still be defined explicitly. |
| The scale factor \(2.38^2/d\) | `roberts1997scaling` | [Roberts, Gelman and Gilks (1997)](https://sites.stat.columbia.edu/gelman/research/published/theory7.pdf) derive an asymptotic random-walk Metropolis scaling under target assumptions. This motivates a convention in the experiments, rather than establishing the optimal scale for every test distribution. |
| SWAP distance-based covariance readout | `lloyd2013distance,buhrman2001fingerprinting` | [Lloyd, Mohseni and Rebentrost (2013), pp. 3–4](https://arxiv.org/pdf/1307.0411), supplies the distance protocol used in `circuit_costs.swap_test`. [Buhrman et al. (2001)](https://arxiv.org/abs/quant-ph/0102001) supplies the underlying SWAP test. A plain overlap-squared SWAP test would lose the sign of a real covariance; the distance protocol recovers it from norm-weighted states. The comparison measures the trace it needs with a separate pairwise-moment circuit rather than supplying it. |
| Hadamard signed-overlap readout | `aharonov2009hadamard` | Section 2.2 of [Aharonov, Jones and Landau](https://arxiv.org/pdf/quant-ph/0511096) describes the real-overlap Hadamard test used by the comparison. Variances measured with pairwise-moment circuits restore the covariance scale. |
| The amplitude-estimation curve currently implemented | `brassard2002amplitude` | [Brassard et al. (2002)](https://arxiv.org/abs/quant-ph/0005055) is the canonical phase-estimation algorithm whose output law `qae_sample` reproduces. The preprint appeared in 2000; the publication year is 2002. |
| Precision scaling for quantum expectation estimation | `montanaro2015montecarlo,cornelissen2022mean` | [Montanaro (2015)](https://arxiv.org/abs/1504.06987) treats Monte Carlo expectation estimation; [Cornelissen, Hamoudi and Jerbi (2022)](https://arxiv.org/abs/2111.09787) is a stronger reference when discussing vector-valued estimates, dimension, and access models. |
| Preparing a correlated Gaussian proposal state | `kitaev2008wavefunction,deliyannis2021gaussian` | [Kitaev and Webb (2008)](https://arxiv.org/abs/0801.0342) gives the construction; [Deliyannis et al. (2021)](https://arxiv.org/pdf/2109.10918) gives explicit circuits and resource bounds. This is the most direct quantum proposal-preparation alternative. The latter's Eq. (34), Appendix E, and Appendix F support the shear costs quoted below. |
| Covariance encoded in a quantum state | `gordon2022covariance` | [Gordon et al. (2022)](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.030334) directly addresses covariance matrix preparation and centering. It is a closer covariance reference than a generic quantum PCA citation. Its output is a covariance representation, rather than a prepared random-walk step. |
| A lower-qubit alternative to canonical amplitude estimation | `grinko2021iterative` | [Grinko et al. (2021)](https://www.nature.com/articles/s41534-021-00379-1) avoids the phase-estimation register. Include this when discussing qubit-efficient alternatives. The plotted canonical curve should retain its canonical label and citation. |
| Quantum walk spectral bounds | `szegedy2004spectra,lemieux2020efficient` | [Szegedy's 2004 preprint](https://arxiv.org/abs/quant-ph/0401053) is titled *Spectra of Quantized Walks and a \(\sqrt{\delta\epsilon}\) Rule*. Its marked-state search bound has specific oracle and chain assumptions. Lemieux supplies the more directly relevant MH circuit construction. |
| Related adaptive quantum sampling | `pyretzidis2026adaptive` | [Pyretzidis, Martínez de Lejarza and Rodrigo (2026)](https://www.nature.com/articles/s42005-026-02684-7.pdf) studies quantum adaptive importance sampling with correlated proposals on a fixed grid. Useful recent context for adaptation and correlated distributions; it trains circuit parameters, whereas our AMR changes the physical window. |

For example, the Gaussian-preparation comparison can cite
`\cite{kitaev2008wavefunction,deliyannis2021gaussian}`, and the SWAP row can
cite `\cite{lloyd2013distance,buhrman2001fingerprinting}`. The Deliyannis entry
follows the author order on the paper's title page; arXiv's abstract page
lists the same authors alphabetically. Quantum PCA is retained separately
as `lloyd2014qpca`, using its 2014 journal publication rather than the 2013
preprint date; it is not the source of the implemented SWAP distance protocol.

Additional entries cover methods already named in the comparison code:
`ledoit2004covariance` for [covariance shrinkage](https://www.ledoit.net/Well-conditioned2004.pdf),
`roberts2009adaptive` for the [adaptive Metropolis safety mixture](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2009.06134),
and `pasarica2010jumped` for [expected squared jumped distance](https://www3.stat.sinica.edu.tw/sstest/j20n1/j20n113/j20n113.html).
Our worst-direction score is a variant of that last objective, not the
paper's original scalar score. `cornelissen2021multivariate` supplies
[further estimation bounds for different access models](https://arxiv.org/abs/2107.03410).
Use these entries where the corresponding methods are discussed.

### Attribution and wording to keep precise

- Describe the comparison as **selected covariance-based step generators**.
  The cited papers establish particular algorithms, not that SWAP tests or
  entrywise estimation are the most widely used ways to generate quantum steps.
- Attribute the no-entry-readout property to our direct step construction.
  Quantum covariance can also be retained in a quantum representation, as in
  Gordon et al.; measurement is a choice of the implemented readout baselines.
- Define the coin sum through its signed pair sum and covariance identity. Its
  Gaussian limit concerns that sum's output distribution. The mathematical
  literature on noncommuting quantum central limit theorems addresses a
  different object and is not a source for a circuit speed advantage here.
- Cite our own window rule and experiments for the AMR criteria, zoom amounts,
  and measured accuracy improvements. Tian's function grid and the QAIS
  sampling grid provide context, not those numerical results. The current
  register mapping is an affine window, so the mesh terminology should be
  defined accordingly.

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
the [walk figures](../README.md) is the expected
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
| Classical pair-difference sampler | Draw the same independent three-outcome pair coins and add their signed history differences | Classical history | \(O(K^2d)\) arithmetic per step with the unoptimized pair list, and exactly the same next-step law as the coin-sum circuit when only its position is measured. It needs no explicit covariance matrix. This is the strongest classical control for any claimed speed advantage. Related history-difference proposals appear in [ter Braak (2006)](https://link.springer.com/article/10.1007/s11222-006-8769-1) and [ter Braak and Vrugt (2008)](https://link.springer.com/article/10.1007/s11222-008-9104-9). |
| Coin-sum pair-difference step, this repository | Coherent coins choose signed pair differences and update a quantum position register | Coherently encoded history, or the six preparatory signs used here | Conditional covariance is exactly \(C_h\) on the rounded lattice. It does **not** make a Gaussian law, remove history sampling error, or by itself accelerate a classical calculation. Width is 29, 33, 41, and 49 logical qubits for 6, 8, 12, and 16 bits per coordinate with one kick. |
| Coin sum + AMR, this repository | Same coin-sum step, with grid center, axes and spacing chosen from final measurements of an earlier pilot circuit | Measured pilot histories between circuits | Same final-circuit width as the coin sum at a fixed bit count. Smaller rounding error is possible, but pilot samples, eigendecomposition, window-safety checks and circuit recompilation cost time. The window update is classical. |
| Correlated Gaussian state preparation | Prepare independent one-dimensional Gaussian states and reversibly shear them into a correlated quantum proposal | Covariance/Cholesky-like parameters, usually classical | Direct next-step comparator, provided a covariance is already available. Finite-grid rounding and state-preparation error matter. Fully history-controlled preparation requires reversible covariance arithmetic and controlled loading not supplied for free. [Kitaev and Webb](https://arxiv.org/abs/0801.0342); [Deliyannis *et al.*](https://arxiv.org/abs/2109.10918). |
| Covariance density matrix and quantum PCA | Amplitude encode centered observations and prepare a density matrix proportional to covariance; diagonalize in a quantum representation | Efficient amplitude access to centered data, copies for PCA | Produces spectral information, **not** a sampled next step. A square-root transform and proposal loader would still be needed. Uncentered data require care ([Gordon *et al.*, 2022](https://arxiv.org/abs/2204.03495); [Lloyd, Mohseni and Rebentrost, 2014](https://arxiv.org/abs/1307.0401)). |
| Quantum expectation or amplitude estimation | Estimate covariance entries, then use a separate Gaussian proposal circuit | Reversible sample oracle and its inverse | Scalar bounded expectations can have \(O(1/\varepsilon)\), rather than \(O(1/\varepsilon^2)\), precision query scaling. Covariance needs up to \(d(d+1)/2\) entries and a proposal loader; oracle preparation, numeric precision, confidence and dimension affect total cost. The estimate is an intermediate classical output, outside the preferred next-step-only scope ([Brassard *et al.*](https://arxiv.org/abs/quant-ph/0005055); [Montanaro](https://arxiv.org/abs/1504.06987); [Cornelissen and Jerbi](https://arxiv.org/abs/2107.03410)). |

Multivariate mean estimation can improve precision query complexity under
particular oracle models, while dimension costs and lower bounds remain
important ([Cornelissen, Hamoudi and Jerbi](https://arxiv.org/abs/2111.09787)).
The many covariance-estimation variants in
`cov_methods.py` therefore do **not** count as direct
next-step baselines here. They are possible components of a longer pipeline.

Quantum Metropolis-Hastings walks address a different layer: they quantize a
proposal and acceptance kernel. They require a suitable proposal oracle and
often substantial reversible arithmetic
([Lemieux *et al.*](https://arxiv.org/abs/1910.01659)). Spectral-gap speedups
for fixed reversible chains cannot simply be assigned to an adaptively
changing AMR walk ([Szegedy](https://arxiv.org/abs/quant-ph/0401053)).
Our current coin-sum circuit leaves its coin/history registers entangled with
the position; it is a coherent proposal experiment, not yet a cleaned
Szegedy transition oracle.

## Published Gaussian preparation cost, specialized to our small case

[Deliyannis *et al.*](https://arxiv.org/pdf/2109.10918) give an explicit
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
Deliyannis *et al.* also observe that generic short-register state loading can be
cheaper than the asymptotically efficient one-dimensional Gaussian loader at
small \(k\), while optimized shear arithmetic becomes useful as the number
of coordinates grows. The coin sum's 33-qubit 8-bit circuit includes its history and
proposal coins, so its width is similarly modest but its arithmetic is deep.
For example, decomposing one local-grid 8-bit coin-sum circuit with Qiskit 1.1.0's
`u,cx` basis and zero optimization gives 57,806 CX gates, 121,873 gates in
total, and depth 69,065. These are compilation diagnostics, not optimized
hardware estimates. The published Gaussian shear CNOT bound omits loading
and history processing, so comparing these two numbers as an end-to-end
speedup would be invalid.

## Matched numerical experiments in this repository

The direct-step comparison (`figures/coin_sum_step_generation.pdf`),
run by [`covariance_methods.py`](../covariance_methods.py), scores all 64
history branches against the **same unrounded physical conditional target**.
The fixed-grid coin-sum output law has 8.38% mean physical covariance error,
with a 0.93% sample standard deviation across history branches. The plotted
readout curves use seven methods: the two Cauchy–Schwarz approximations, SWAP,
Hadamard, canonical amplitude estimation, covariance preparation followed by
readout, and pairwise moments.
Each uses the same rounded histories, continuous-history target, PSD
projection, and ideal downstream Gaussian loader. Explicit small circuits
validate the measurement probabilities and the prepared covariance state;
the numerical experiment samples finite-shot outcomes from these laws.

No method receives exact variances, norms or traces. SWAP and Hadamard measure
the column variances with pairwise-moment circuits. Covariance preparation
prepares the norm-weighted ensemble whose density matrix is proportional to
the raw sample covariance, and measures the trace with a pairwise-moment
circuit on squared pair distances. Its
finite-shot matrix readout is an added comparison stage, not a step supplied
by the QPCA paper. This added stage gives nonzero finite-shot error. The
[figure guide](../README.md) documents the full protocol.

The two coin-sum walk circuits use 33 logical qubits. The implemented covariance
stages use 5 qubits for SWAP, 3 for Hadamard, 5 for pairwise moments, 11 for
canonical amplitude estimation, and 3 for the purified covariance ensemble.
Readout methods still need a Gaussian step loader with at least 16 output
qubits; those stage counts exclude loader workspace and coherent history
access. The resource table uses only the same ten methods as the error table.

QAE now counts both preparation and inverse-preparation calls. At the total
384-call budget, three entries each use 64 phase states and 127 calls,
for 381 calls overall. Its previously reported 128 phase states per entry
would require 765 calls overall. Finite phase resolution can give
nonmonotonic errors between successive budgets; it does not remove the
finite-grid floor. The covariance-preparation readout measures two bases
with 192 shots each at the 384-shot budget.

The AMR covariance-error comparison (`figures/coin_sum_amr_covariance_error.pdf`) and
AMR pilot-reliability comparison (`figures/coin_sum_amr_pilot_success.pdf`), run by
[`amr.py`](../amr.py), holds the final coin-sum circuit at
8 bits per coordinate and 33 logical qubits. It compares the full physical
range, an informed axis-aligned local window with the same pilot width, and
the measured principal-axis AMR window. Seven step geometries probe tilt,
sign, isotropy, scale, and two fixed-grid-friendly regimes. For each geometry, 80 independent subsets of
pilot final measurements are used at each of seven pilot budgets. Errors
are exact circuit-law covariance errors relative to the continuous target;
unsafe windows count as unsuccessful trials. One separate 30,000-shot
circuit-law check per geometry validates the analytic scoring.

| Geometry | Full-range 8-bit error | Informed local 8-bit error | Untuned AMR 8-bit mean ± SD, 1,000 history samples |
| --- | ---: | ---: | ---: |
| Tilted | 169.9% | 9.04% | 3.35% ± 2.06% |
| Axis-aligned | 264.8% | 11.29% | 3.69% ± 1.15% |
| Anti-tilted | 169.9% | 9.04% | 3.12% ± 1.94% |
| Isotropic | 64.1% | 8.74% | 5.04% ± 1.99% |
| Broader | 16.9% | 9.04% | 3.31% ± 1.82% |
| Grid-aligned | 152.8% | 0.114% | 5.23% ± 1.85% |
| Isotropic, already resolved | 157.1% | 0.083% | 5.19% ± 2.15% |

At 1,000 history samples, the tilted AMR error is 3.35% ± 2.06%, where the
spread is one sample standard deviation across 80 selected windows. The
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

The earlier `Kitaev–Webb-style replica` prepares two
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
it is handed a covariance that the coin-sum circuit computes coherently from
its history. It also targets a Gaussian law whereas the coin sum targets the same
second moment with a discrete pair-difference law. The replica's 5.83%
rounding error uses the **lattice** covariance denominator; the AMR table
uses a **continuous physical** denominator. The two percentages should not
be read as a head-to-head ranking. An end-to-end Gaussian comparison must
include reversible history covariance computation and covariance-controlled
state preparation, or measure/compute covariance classically and be labelled
hybrid.

The AMR gain itself is a **grid-resolution** effect and is not unique to
the coin sum: a Gaussian step sampled from the same rounded history covariance on
the same physical window inherits essentially the same physical covariance
rounding error. The coin sum's distinguishing feature is how it produces the
history-conditioned second moment without first materializing covariance
entries. Neither this stress test nor the Gaussian replica demonstrates a
speed advantage over classical adaptive sampling.

The explicit 6-bit lookup circuit is deliberately simple and expensive:
Qiskit 1.1.0 zero-optimization `u,cx` transpilation gives 748,520 CX gates,
1,499,445 gates in total, and depth 1,242,854 on the tilted case. This is
**not** Deliyannis *et al.*'s efficient shearing implementation.
It validates the output law and illustrates why high-level `StatePreparation`
or controlled additions cannot be counted as unit-cost hardware gates.

## Selected covariance-method comparison

[`covariance_methods.py`](../covariance_methods.py) generates the
method bar chart and full error table from the ten-method list in
[`covariance_methods.py`](../covariance_methods.py):

1. The coin sum on the fixed informed local grid.
2. The coin sum with AMR.
3. SWAP test.
4. Hadamard test.
5. Canonical amplitude estimation.
6. Covariance matrix preparation, with finite-shot readout added.
7. Cauchy–Schwarz upper bound.
8. Cauchy–Schwarz upper/lower bound average.
9. Single history-pair difference.
10. Pairwise moments.

The bound average is exactly the diagonal approximation: the two signed
cross-term bounds cancel while the diagonal variances remain. It is
included once under that requested name. Both bound approximations measure
the variances with pairwise-moment circuits, within the same 384-shot budget,
and feed the same ideal Gaussian loader as the other readout methods. The single-pair step is a direct signed difference, rounded to the
same local grid; it is related to differential-evolution proposals.

The bar chart reports mean error divided by the mean coin sum + AMR error,
with one sample standard deviation on the same scale. The full table gives
mean and sample SD in percent, using unrounded individual errors.
Independent classical covariance, shrinkage, uncentred second moments,
isotropic proposals, the lower bound alone, and the window-only proposal
are no longer entries in the current comparison. Earlier stored-history
experiments are separate from the paper figures.

## What the evidence supports

The clean finding is a resource/accuracy trade: a 33-logical-qubit coherent
history-dependent coin-sum proposal can represent the intended covariance much
better when a measured pilot selects an appropriate physical grid. For five
controlled geometries, the improvement survives repeated pilot draws and
an informed local-grid baseline; for two well-resolved geometries AMR is
substantially worse. The quantum circuit supplies a step,
not an extracted covariance matrix. A practical advantage over a classical
adaptive Gaussian or pair-difference sampler, and a quantum mixing-time
speedup, remain unestablished. Classical sampling from the same small coin
law is straightforward; the coin sum's value as a proposal subroutine would require
an application where coherent access to the step and subsequent interference
are actually used.

## Reproduce

The [project README](../README.md) gives the code reading order and dependencies.
From the repository root:

```bash
python3.12 -m unittest discover
python3.12 generate_paper_figures.py
python3.12 generate_paper_figures.py --plot-only
```

The first figure command recomputes the six current datasets; the second
redraws them. Output is written directly to `figures/`. Older
Gaussian-replica and stored-history experiments discussed above are historical
records, and their source can be recovered from Git history.
