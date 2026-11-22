# Handoff: what was wrong with the earlier “quantum” framework

Historical source filenames in this note refer to the framework before the
November 2026 cleanup; those files are in the git history. The covariance-scaled
coin sum ("coin sum" below) was called QCLT at the time. The current implementation and reading order are in
[the project README](../README.md).

Read this before modifying the coin sum, AMR, MCMC, or figure code. **The old
application plots do not demonstrate a fully coherent quantum
APT + TD + RJ + MCMC algorithm or a quantum speedup.** They are useful classical MCMC
experiments with an exactly simulated circuit-compatible proposal law. A
separate, much smaller study now builds actual coherent coin-sum step circuits.

**Five points to keep in mind:**

1. The plotted APT/TD/RJ/MCMC runs execute classically; they sample a
   circuit-compatible coin-sum proposal law in NumPy.
2. The 95-qubit and 33-qubit results have restricted input models and do not
   fit the full multiwavelet sampler into 100 qubits.
3. Exact covariance on a rounded grid is not zero error against the intended
   physical covariance, nor proof of better mixing or a quantum speedup.
4. AMR is chosen classically between circuits. It helps some grids and hurts
   already well-resolved ones, while adding pilot and recompilation costs.
5. Several favorable old comparisons relied on flawed or mismatched
   benchmarks. Their corrections and remaining limits are recorded below.

## Start with the execution boundary

| Code/result | What actually runs | What it establishes |
| --- | --- | --- |
| `tdrjmcmc_qclt.py`, `qclt_amr.py`, and old application figures (in git history) | Classical NumPy MCMC. Coin-sum pair coins and steps are sampled classically. Likelihood, acceptance, birth/death, tempering, history, grid updates, and waveform evaluation are classical. | Performance of a *classical sampler using the coin-sum proposal distribution*. A separate circuit check verifies that the proposal law agrees with a Qiskit circuit. |
| `qcov.py` and `compact_qclt.py` | Qiskit proposal circuits and exact ideal output-law samplers. | One coherent kick or a small coherent walk; no end-to-end MCMC. |
| `quantum_walk_study.py` and focused walk figures (in git history) | A two-coordinate, three-history coherent walk segment, with no intermediate measurement inside a segment. | History-conditioned step generation. AMR is selected **classically between separate circuits** from pilot measurements. No likelihood or Metropolis acceptance. |
| `qaptdrjmcmc.ipynb` | A partial circuit with a placeholder likelihood. | It does not implement the full plotted sampler. |

Calling the old application methods “quantum” without the words **“coin-sum
kick distribution simulated classically”** is misleading. The coin-sum circuit
and its classical coin sampler have the same final measurement law here
because the circuit uses basis-preserving arithmetic after preparing the
coins; no useful interference or quantum speedup has been demonstrated.
The circuit equivalence check is valuable, but it does not turn a classical
MCMC execution into a coherent quantum sampler.

## Problems with resource claims

1. **A kick was confused with an algorithm.** The initial qubit counts covered
   history registers, a signed step output, and proposal coins for one
   wavelet. They omitted quantum registers and reversible logic for the
   current multiwavelet state, likelihood, Metropolis decision, reversible
   jumps, eight temperatures, and AMR. For six 5-parameter histories, the
   original history-loaded construction needs **615** logical qubits for
   the pair kick, or **625** including coherent isotropic coins, at 16 bits.
   At 6 bits the analogous counts are **265/275**. Earlier **610/260**
   numbers used a signed output register one bit too short; see the
   [qubit audit](QUBIT_AUDIT.md).
2. **The sub-100 result has a different input model.** When the history is a
   classical NumPy array, `compact_qclt.py` can compile
   its pair differences into gate constants and omit all quantum history
   registers. The resulting 6-bit AMR kick is at most **95 qubits** including
   isotropic coins, with the same proposal law. It must be recompiled if the
   history or window changes. This is a coherent *step accumulator fed by
   classical data*, not a coherent MCMC history. The original sampler
   already samples the same coins entirely classically.
3. **The 29–33-qubit coherent walk is a special toy construction.** Its
   histories are deterministic functions of six still-coherent preparatory
   signs and known displacement modes. It computes pair differences from
   those signs without storing snapshots. This exact compression does not
   apply to arbitrary unknown quantum history or branch-dependent
   likelihood/acceptance trajectories. The 8-bit two-coordinate circuit
   has **33** logical qubits; a five-coordinate, six-history, 6-bit version
   of the same construction would already use **101** including its
   isotropic floor. The formula and assumptions are in
   [QUBIT_AUDIT.md](QUBIT_AUDIT.md).
4. **A tagged superposition does not provide two values to subtract.** The
   state `|psi_0>|0> + |psi_1>|1>` uses one data register, but the pair
   difference needs both values accessible in the same branch. Classical
   values can instead be loaded or compiled as constants, trading width for
   gates. Unknown independent quantum inputs cannot generally be compressed
   into one state register plus one tag without losing the joint access the
   operation needs.
5. **The full application exceeds the requested width before oracles.** Even
   an information-efficient encoding of one 10-wavelet, five-parameter,
   16-bit configuration needs about **779 state qubits**; eight such chains
   need about **6,226**. These lower bounds exclude proposals and all
   oracles. The production AMR sampler keeps its *state* on the 16-bit
   grid; the 6-bit window applies only to a kick. Calling the whole
   eight-chain problem “under 100 qubits” is untenable without changing the
   problem or the meaning of coherence.
6. **Logical width hid circuit depth and hardware cost.** High-level Qiskit
   `mcx`, additions, and `StatePreparation` are not unit-cost hardware
   gates. A zero-optimization `u,cx` decomposition of one 33-qubit 8-bit
   local coin-sum step has **57,806 CX gates** and depth **69,065**. No noise,
   connectivity, fault-tolerance, or physical-qubit estimate has been
   demonstrated. Fresh proposal coins are needed for independent coherent
   steps; reusing an unreset coin does not create a fresh draw.

## Problems with the covariance and “better estimator” claim

- **The requested output is the next random step, not a measured covariance
  matrix.** Pairwise-moment, SWAP-overlap, quantum PCA, and amplitude
  estimation routines often output covariance information. They need an
  additional step loader before they solve the same task as the coin sum. Conversely,
  classical pair-difference sampling produces exactly the same measured
  coin-sum step law with little arithmetic. Do not equate “fewer covariance
  readout shots” with a general quantum speedup.
- **The old 0% coin-sum error used the wrong target for a physical comparison.**
  The coin sum has exactly the prescribed conditional covariance *of its rounded
  lattice history*. This is an algebraic identity, not proof that it matches
  the intended continuous physical covariance. The corrected
  step comparison (`figures/coin_sum_step_generation.pdf`) scores
  every method against the same unrounded, history-specific target across
  64 branches. On an 8-bit local grid, the coin sum has **8.8% median physical
  error**. At 384 intermediate covariance-measurement shots, ideal-loaded
  pairwise and SWAP pipelines have **11.8%** and **9.3%** median error; their
  curves approach the same grid-error floor as shots increase. The AMR plot
  instead scores the *unconditional* target, hence its local-grid **9.04%**
  number. These are different summaries of the same rounding problem.
- **There was no equal-qubit comparison.** The corrected figure shows **33
  qubits** for the complete known-mode coin-sum step, versus **5 qubits** for
  each pairwise or SWAP *entry-readout* circuit. Those 5-qubit circuits do
  not produce a step. A two-coordinate 8-bit Gaussian output register
  alone needs **16** state qubits; the published Kitaev–Webb shear stage
  uses **31** including its arithmetic ancillae, with covariance supplied
  classically and one-dimensional loading ancillae omitted. A complete
  coherent history-conditioned estimator-plus-loader circuit was not built
  or costed. The measured-entry curves are optimistic: they receive a
  classical history branch and a free exact loader; SWAP also receives
  exact variances. They are analytic shot models checked against actual
  5-qubit circuit probabilities, not hardware benchmarks.
- **The coin sum's proposal law is discrete and can differ from adaptive Gaussian
  Metropolis even when covariance agrees.** In the application code it uses
  only six reservoir histories for five coordinates, so the covariance is
  noisy. The probability `q` of using each pair is capped at 1; when the
  adapted scale is larger, the coin sum's step covariance no longer follows the
  intended scaled covariance. The isotropic lattice floor, rounding,
  periodic-phase convention, and acceptance rate can all affect mixing.
  Equal conditional covariance does not imply equal posterior efficiency.

See the [literature and matched-test review](LITERATURE_REVIEW.md) for
the access assumptions and costs of the other methods. It does **not**
establish that the coin sum dominates all common covariance estimators.

## Problems with AMR claims

- **AMR is classical between circuits.** The pilot's final samples set the
  next center, axes, and spacing via NumPy. A circuit then runs coherently
  under that fixed window. It is incorrect to call this a fully coherent
  in-circuit AMR update or a coherent quantum MCMC.
- **The qubit saving concerns a representation, not total runtime.** At the
  same 33-qubit final width, AMR can reduce physical grid-rounding error,
  but it also consumes pilot executions, PCA, window checks, and circuit
  recompilation. The gain is not unique to the coin sum: another proposal family
  on the same improved grid can also benefit.
- **AMR is not uniformly better.** The seven-geometry stress
  test (`figures/coin_sum_amr_covariance_error.pdf`) compares AMR and fixed grids on seven cases, while
  the pilot reliability plot (`figures/coin_sum_amr_pilot_success.pdf`) reports the run-level outcomes. The comparison finds cases where
  AMR beats an informed local fixed grid and two where it loses. On the
  tilted case, local fixed error is **9.04%** versus **2.61%** AMR median;
  on a grid-aligned case, it is **0.114%** versus **4.88%**. When fixed
  spacing already resolves modes almost exactly, AMR rotation/re-spacing
  can add error. Small pilots can also misplace a window or make it unsafe.
  The original **0.955%** AMR result came from one favorable pilot; the
  repeated-pilot tilted median is **2.61%**.
- **An attempted secondary tuning rule was invalid.** It optimized the
  final window to reproduce the 6-bit pilot's *step covariance*, which was
  already **50.9%** off the continuous target. This rewarded preserving
  pilot-grid bias and consistently worsened the final result. That tuning
  was removed; the current plots use only the measured-history window rule.
- **The production AMR sampler has extra effects.** It clips stored
  reservoir points that fall outside the moving window and rejects moves
  leaving the window. Those choices can affect mixing. The general
  `qclt_amr.py` default `K=2d` differs from the five-parameter
  wavelet sampler's `K=d+1`: at `d=5`, ten histories mean 45 pairs and 90
  pair coins, already near 100 qubits before output registers.

## Earlier notebook and circuit traps

- `FINAL.ipynb` computes covariance and its matrix square root classically;
  a small quantum-walk primitive there is not a coherent implementation of
  the adaptive covariance calculation.
- `qapthtdrjmcmc.ipynb` has a 10-data-qubit, 5-coin-qubit walk on four-point
  grids. It reuses the same coins over two steps, so the steps are not fresh
  independent draws. On a four-point periodic grid, its `+2` and `-2`
  moves also coincide for four of five parameters.
- `qaptdrjmcmc.ipynb` allocates 230 qubits for a five-wavelet, three-bit
  example; two chains plus one copied history account for 225. Its
  likelihood is marked as a placeholder and it lacks Metropolis
  acceptance, birth/death RJ maps, and AMR. The width is not evidence for a
  complete coherent sampler. `qcov.build_adaptive_walk` is a
  separate coherent proposal demonstration, whose snapshots and fresh
  coins grow with the number of steps.
- The full-grid proposal in `tdrjmcmc_qclt.py` uses raw
  phase differences across the periodic seam, while the generic
  `qclt_amr.py` uses minimal-image differences. This can
  make full-grid phase kicks unnecessarily large near the seam. The
  compact-circuit equality test preserves each sampler's existing
  convention; it does not resolve that modeling difference.

The early covariance notebooks and `amelia.py` also had
statistical and bookkeeping problems. The corrected
`cov_methods.py` documents them: an inner product
`trace(C_est^H C_true)` was called an error; a zero-mean covariance estimate
missed its `1/(N-1)` normalization; two independently generated datasets
were used to test a cross covariance whose truth was then zero; a weighted
sum was described as an 80/20 random mixture; a purported uniform phase
was generated by exponentiating a uniform log value; raw parameter scales
differed by roughly 20 orders of magnitude; and large generated waveforms
were unused. Do not reuse the old scores or notebook plots as evidence for
estimator accuracy. Even the corrected covariance-entry comparisons need
the missing next-step loader and matched input assumptions before they can
support an end-to-end proposal claim.

## Problems found in the old application figures

The old figure guide (in git history) documents a later audit
of experimental fairness. Several defects were **fixed in the current
archived plots**, so do not report them as still-active code bugs:

| Earlier issue | Why the earlier conclusion was unreliable | Current status |
| --- | --- | --- |
| Our repeated runs shared one favorable noise realization while BayesWave runs had different ones. | Run-to-run spread and cross-sampler comparisons were optimistic. | Cross-sampler studies now vary noise by seed; single-posterior studies intentionally hold it fixed and say so. |
| The 8,192 waveform evaluations used to build a time-frequency birth map were omitted. | TF-birth methods appeared cheaper. | Setup evaluations are now counted. |
| “Reaches 0.95” meant one instantaneous crossing. | It did not match a sustained posterior-accuracy metric. | Now uses the median over the second half of the run so far. |
| Fisher derivatives were costed by finite differences even though a closed form is available for these wavelets. | A claimed fourfold coin-sum cost advantage over Fisher was an artifact. | The claim was withdrawn; plots show actual Fisher and a free-derivative reference. |
| An adaptive-grid wavelet walk started near the injected truth. | The test had an unrealistic head start. | Start now comes from the data's time-frequency map; an adaptive 16-bit baseline was added. |
| One adaptive-grid test treated phase as non-periodic. | It gave misleading behavior at the phase seam. | Periodic phase, circular statistics, and wrapped references were added. |

Remaining comparison limits: BayesWave and this toy sampler use different
priors, noise, parameter ranges, and numbers of temperatures; BayesWave
evaluation counts are approximate; and single-core Python-vs-compiled-C
timing measures implementations. Counting a classical waveform evaluation
as a future “quantum oracle call” does **not** account for constructing a
reversible waveform/likelihood oracle or the quantum proposal and
acceptance machinery. Do not turn the old application cost plots into
quantum runtime or query-complexity claims.

## What remains open; how to work on this code

- A fully coherent APT + TD + RJ + MCMC with AMR still needs a reversible
  likelihood/waveform oracle, an acceptance construction, birth/death and
  temperature transitions, history handling, and a coherent or explicitly
  hybrid AMR policy. `qaptdrjmcmc.ipynb` has a placeholder likelihood and
  lacks these components. Neither the compact kick nor the 33-qubit walk
  supplies them.
- Keep three resource scopes separate in code and figures: **(1)**
  covariance-entry readout, **(2)** a complete next-step proposal with its
  input model, and **(3)** a complete sampler. Count logical qubits, fresh
  coins, ancillae, decomposed gates/depth, circuit executions, classical
  updates, and oracle work as applicable. Label whether a history is
  classical, prepared from known signs, or an arbitrary coherent input.
- Score methods against the same **physical** target and the same output
  task. State whether the metric is conditional or averaged over histories,
  and whether the comparison includes sampling noise, grid error, and
  loader/oracle costs. Show AMR failure regimes, not only favorable pilots.
- Preserve the useful tests: `test_compact_qclt.py`
  checks exact kick-law equality, and
  `test_quantum_walk_study.py` checks the
  compressed coherent history against explicit snapshots. Reproducing the
  focused figures (in git history) additionally runs
  `qclt_step_comparison.py` and `amr_battle_test.py`.

The shorter [project README](../README.md) points to current results. This
handoff records the issues so they are not reintroduced while changing the
framework.
