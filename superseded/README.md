# Superseded

First version of the wavelet TD-RJMCMC comparison, kept for reference only. It was misnamed
"BayesWave". It is a BayesWave-style Python proxy, not LIGO's BayesWave.

- `wavelet_rjmcmc_v1_reservoir.py` (was `bayeswave_qclt.py`): adaptive kernels used a per-wavelet
  reservoir of past states, and PT used reversible random-pair swaps.
- `bayeswave_qclt.*`: its 20k-iteration results. The adaptive and QCLT runs had not converged in model
  dimension at 20k iterations. An 80k-iteration check (`_long_check.*`) showed ⟨K⟩ drifting from 6.7
  down to the Fisher value (~5.3), i.e. slow mixing rather than bias.

Replaced by `../tdrjmcmc_qclt.py` (Tian et al. AP-PT-RJMCMC adaptation) and `../tdrjmcmc_figures.py`.
`_long_check.py` imports the old module name and will not run as-is.

## Cost benchmark and t0 scan, first versions (replaced by `../apt_study.py`)

- `cost_benchmark.*`, `tdrjmcmc_t0_sweep.*`: the first per-evaluation cost benchmark and t0 scan.
  The honesty audit found four problems, all fixed in `apt_study.py`:
  1. every run of ours used the same noise realisation (`data_seed = 1`), while every BayesWave run had its
     own;
  2. the 8192 waveform evaluations spent on the time-frequency birth map were not counted;
  3. "reaches 0.95" was the first iteration whose instantaneous match touched 0.95, an optimistic
     definition not comparable with BayesWave's posterior median; now the median over the second half;
  4. Fisher's cost was reported only with finite-difference derivatives (about 20 waveform evaluations per
     move). For sine-Gaussian wavelets the Fisher matrix is available in closed form (BayesWave uses an
     analytic one), so Fisher is now also shown "if derivatives were free".
  The headline "QCLT-AP needs about 4x fewer evaluations than Fisher" rested on item 4 and does not hold
  against a Fisher sampler with analytic derivatives.
- `tdrjmcmc_t0_sweep_v2_log.txt` is empty: that v2 run was stopped once the problems above were found.
- These scripts write into `../figures` and `../data` and import from the parent directory; they will not
  run from here as-is.
