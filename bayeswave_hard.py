"""Real BayesWave on the hard case (wide chirp + extra burst) at SNR 40, for the cost and t0 comparisons.

Reuses snr_sweep.py's BayesWave job (run + post-processing at checkpoints) with the hard-case injection
written by `bayeswave_compare.py make-mdc <psd> ~/bayeswave/work/hard/mdc hard`.

Run:   python3.12 bayeswave_hard.py     -> data/bayeswave_hard.pkl
"""
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import snr_sweep as W

W.WORK = Path.home() / "bayeswave" / "work" / "hard"
W.MDC_CACHE = W.WORK / "mdc" / "H1_mdc.cache"
SNR, SEEDS = 40, 4

if __name__ == "__main__":
    t = time.time()
    with Pool(4) as pool:
        res = pool.map(W._bw_job, [(SNR, s) for s in range(SEEDS)], chunksize=1)
    print(f"BayesWave hard case: {len(res)} runs in {time.time() - t:.0f}s; injected SNR "
          f"{[round(r['inj_snr'], 2) for r in res]}", flush=True)
    W.DATA.mkdir(exist_ok=True)
    with open(W.DATA / "bayeswave_hard.pkl", "wb") as f:
        pickle.dump(res, f)
