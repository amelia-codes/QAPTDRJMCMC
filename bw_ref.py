"""Real-BayesWave reference data for the comparison figures (loaded from data/).

BayesWave's units differ from ours in two ways, handled here:
  * one BayesWave iteration = 100 model updates per chain (ours: 1), so x-axes use "model updates per chain";
  * its match ("overlap") is the median over the posterior samples saved in the second half of the chain up
    to a checkpoint (from BayesWavePost), not the instantaneous cold-chain match we record every iteration.
Waveform evaluations are approximated as model updates x chains, i.e. one waveform/likelihood evaluation per
model update. This is an approximation, not a bound: updates rejected by the prior need none, while extrinsic
and noise-model updates (not counted as model updates) need more.

Two BayesWave configurations:
  "default": BayesWave's defaults (20 chains, fits the noise spectrum, samples sky/polarisation)
             -> data/snr_sweep.pkl (short chirp) and data/bayeswave_hard.pkl (hard case)
  "matched": 8 chains, noise spectrum fixed, sky fixed at zenith (bayeswave_matched.py)
             -> data/bayeswave_matched.pkl
  "matched_fixedT": as "matched", temperature ladder not adapted (bayeswave_matched.py --fixed-ladder)
             -> data/bayeswave_matched_fixedT.pkl
"""
import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
UPDATES_PER_IT = 100
N_CHAIN = 20
LABEL = "BayesWave (real, classical)"
LABELS = {"default": "BayesWave, default settings (classical)", "matched": "BayesWave, matched settings (classical)",
          "matched_fixedT": "BayesWave, matched settings, fixed temperature ladder (classical)"}
COLOR = "#0b0b0b"


def runs(case, variant="default"):
    """BayesWave runs: case = SNR (10/20/40/80) on the short chirp, or "hard" (wide chirp + burst, SNR 40)."""
    if variant in ("matched", "matched_fixedT"):
        p = DATA / ("bayeswave_matched.pkl" if variant == "matched" else "bayeswave_matched_fixedT.pkl")
        if not p.exists():
            return []
        key = "hard40" if case == "hard" else f"chirp{case}"
        with open(p, "rb") as f:
            return [r for r in pickle.load(f) if r["case"] == key]
    if case == "hard":
        p = DATA / "bayeswave_hard.pkl"
        if not p.exists():
            return []
        with open(p, "rb") as f:
            return pickle.load(f)
    with open(DATA / "snr_sweep.pkl", "rb") as f:
        return [r for r in pickle.load(f)["bw"] if r["snr"] == case]


def curves(case, variant="default"):
    """Per run: dict(updates, evals, cpu, match) at its checkpoints (match = posterior median overlap)."""
    out = []
    rr = runs(case, variant)
    rates = [r["sec_per_it"] for r in rr if np.isfinite(r.get("sec_per_it", np.nan))]
    for r in rr:
        its = np.array(sorted(k for k, v in r["checkpoints"].items() if v is not None and len(v)), dtype=float)
        if not len(its):
            continue
        spi = r.get("sec_per_it", np.nan)
        spi = spi if np.isfinite(spi) else (np.median(rates) if rates else np.nan)
        nchain = r.get("nchain", N_CHAIN)
        out.append(dict(updates=its * UPDATES_PER_IT, evals=its * UPDATES_PER_IT * nchain, cpu=its * spi,
                        match=np.array([np.median(r["checkpoints"][int(k)]) for k in its]), inj_snr=r["inj_snr"]))
    return out


def median_curve(case, variant="default"):
    """Median over runs at the checkpoints all runs share."""
    cs = curves(case, variant)
    if not cs:
        return None
    n = min(len(c["updates"]) for c in cs)
    return {k: np.median(np.array([c[k][:n] for c in cs]), axis=0) for k in ("updates", "evals", "cpu", "match")}


def first_hit(case, thresh, key, variant="default"):
    """Per run, the `key` value (updates / evals / cpu) at which the median match first reaches thresh."""
    out = []
    for c in curves(case, variant):
        ok = np.nonzero(c["match"] >= thresh)[0]
        out.append(c[key][ok[0]] if len(ok) else np.nan)
    return np.array(out)


def details():
    """SNR-20 seed-0 run: chain wavelet parameters and the post-processed reconstruction."""
    p = DATA / "bayeswave_snr20_s0_details.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


WORK = Path.home() / "bayeswave" / "work"


def wavelet_counts(case, variant="default"):
    """Per run: (model updates per chain, number of wavelets) from BayesWave's saved chain samples."""
    if variant == "matched":
        dirs = sorted((WORK / "matched").glob(("hard40" if case == "hard" else f"chirp{case}") + "_s[0-9]"))
    else:
        d = WORK / "hard" if case == "hard" else WORK / "snr_sweep"
        dirs = sorted(d.glob(f"bw_snr{40 if case == 'hard' else case}_s*"))
    out = []
    for run_dir in dirs:
        f = run_dir / "chains" / "signal_params_h0.dat.0"
        if not f.exists():
            continue
        D = []
        for line in open(f):
            tok = line.split()
            n = int(float(tok[0]))
            D.append(n if len(tok) == 7 + 5 * n else np.nan)       # skip malformed (partially written) lines
        D = np.array(D, dtype=float)
        out.append((np.arange(len(D)) * 100.0 * UPDATES_PER_IT, D))
    return out
