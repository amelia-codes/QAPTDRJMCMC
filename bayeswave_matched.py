"""Real BayesWave with settings matched to our samplers, to separate "BayesWave's settings" from "BayesWave's
sampler" in the comparisons.

Default BayesWave (snr_sweep.py, bayeswave_hard.py): 20 chains, 100 model updates per iteration, fits the noise
spectrum, samples sky position and polarisation. Matched here:
  * 8 chains (ours: 8 temperatures)
  * --noPSDfit: noise spectrum held fixed (ours: known white noise)
  * --fixSky at H1's zenith for the injection time (best antenna response; a single detector cannot measure
    the sky position anyway)
Everything else (wavelet model, priors, proposals, adaptive temperatures, 100 updates per iteration) is
BayesWave's own.

Why not --Ncycle 1 (one update per iteration, like ours)? Timed on 1000 iterations with 8 chains: default 2.1 s,
--Ncycle 1 83 s. BayesWave does an expensive step once per iteration (~0.08 s) that its 100 cheap model updates
(~20 us each) normally amortise, so --Ncycle 1 would be ~40x slower per update and not representative.
BayesWave saves a chain sample every 100 iterations (1e4 updates), so its curves cannot start earlier than
that; the 200- and 300-iteration checkpoints rest on 1-2 samples and are noisy.

Tempering ablation (--fixed-ladder): the same matched settings plus --noAdaptTemperature, i.e. BayesWave's own
initial geometric ladder (spacing 1.2: temperatures 1 ... 1.2^7 = 3.6 for 8 chains) kept fixed, against the
adaptive runs above, which start from that ladder and tune it. This mirrors our ablation (our fixed ladder is
our own initial geometric ladder).

Run:   python3.12 bayeswave_matched.py --test           # short timing run
       python3.12 bayeswave_matched.py                  # 3 cases x 4 seeds -> data/bayeswave_matched.pkl
       python3.12 bayeswave_matched.py --fixed-ladder   # 2 cases x 4 seeds -> data/bayeswave_matched_fixedT.pkl
"""
import pickle
import re
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import bayeswave_compare as BC
import snr_sweep as W

WORK = Path.home() / "bayeswave" / "work" / "matched"
DATA = Path(__file__).parent / "data"
ZENITH_RA, ZENITH_DEC = 0.915180, 0.810795            # H1 zenith at GPS 900000000 (lal; F+^2 + Fx^2 = 1)
NCHAIN = 8
NITER = 200_000
CHECKPOINTS = [200, 300, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
CASES = {"chirp20": (20, Path.home() / "bayeswave/work/snr_sweep/mdc/H1_mdc.cache"),
         "chirp40": (40, Path.home() / "bayeswave/work/snr_sweep/mdc/H1_mdc.cache"),
         "hard40": (40, Path.home() / "bayeswave/work/hard/mdc/H1_mdc.cache")}
EXTRA = ["--noPSDfit", "--fixSky", "--fixRA", str(ZENITH_RA), "--fixDEC", str(ZENITH_DEC)]


def fresh_dir(path):
    """Create an empty run directory. If a previous attempt left this exact directory, remove only it."""
    path = Path(path).resolve()
    assert path.parent == WORK.resolve(), f"refusing to touch {path}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def job(args):
    case, seed, niter, checkpoints = args
    snr, cache = CASES[case]
    d = fresh_dir(WORK / f"{case}_s{seed}")
    bw_args = BC.bw_args(cache, snr * W.SNR_CAL, seed, niter, NCHAIN, extra=EXTRA)
    t0 = time.time()
    rc = BC.bw_run("BayesWave", bw_args, d, d / "bw.log")
    wall = time.time() - t0
    log = (d / "bw.log").read_text()
    m = re.findall(r"Injected network SNR ([\d.]+)", log)
    lines = W._chain_lines(d)
    every = niter / max(lines, 1)                       # iterations between saved chain samples
    res = dict(case=case, snr=snr, seed=seed, rc=rc, wall=wall, sec_per_it=wall / niter,
               inj_snr=float(m[-1]) if m else np.nan, n_lines=lines, save_every=every,
               updates_per_it=100, nchain=NCHAIN, extra=list(EXTRA), checkpoints={})
    W.BW_EVERY = int(round(every))
    for it in checkpoints:
        if it <= niter:
            res["checkpoints"][it] = W._bw_post_truncated(d, bw_args, it)
    return res


if __name__ == "__main__":
    out_name = "bayeswave_matched.pkl"
    cases = list(CASES)
    if "--fixed-ladder" in sys.argv:                  # globals set before the pool forks its workers
        WORK = WORK.parent / "matched_fixedT"
        EXTRA = EXTRA + ["--noAdaptTemperature"]
        out_name, cases = "bayeswave_matched_fixedT.pkl", ["chirp40", "hard40"]
    WORK.mkdir(parents=True, exist_ok=True)
    if "--test" in sys.argv:
        r = job(("chirp20", 99, 20_000, [2_000, 20_000]))
        print({k: (np.round(np.median(v), 3) if v is not None else None) for k, v in r["checkpoints"].items()},
              f"wall {r['wall']:.1f}s, lines {r['n_lines']}, saved every {r['save_every']:.0f} iterations, "
              f"injected SNR {r['inj_snr']}")
        sys.exit()
    t = time.time()
    jobs = [(c, s, NITER, CHECKPOINTS) for c in cases for s in range(4)]
    with Pool(4) as pool:
        res = pool.map(job, jobs, chunksize=1)
    print(f"{len(res)} matched BayesWave runs in {time.time() - t:.0f}s", flush=True)
    DATA.mkdir(exist_ok=True)
    with open(DATA / out_name, "wb") as f:
        pickle.dump(res, f)
