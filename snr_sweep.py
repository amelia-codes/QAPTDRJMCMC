"""Which sampler does better at which signal strength? Real BayesWave vs our TD-RJMCMC samplers.

Same chirp (tdrjmcmc_qclt.make_data, DATASET="chirp") at several signal-to-noise ratios (SNR):
  * BayesWave 1.1.3 (LIGO, conda-forge), simulated Advanced LIGO noise, single detector, signal model only,
    the chirp injected coloured by the noise spectrum so its whitened shape is the same as ours
    (bayeswave_compare.py). One long run per (SNR, seed); BayesWavePost is re-run on truncated copies of
    the chain to get the accuracy reached after a given number of iterations.
  * Our samplers (Fisher, AP with time-frequency birth, QCLT-AP with time-frequency birth; adaptive step
    starting at iteration 1000) on white noise with the same whitened chirp. As in Tian et al., AP and QCLT-AP
    keep the conventional (here: Fisher) move before iteration 1000 and on 20% of moves afterwards.
    Every run has its own noise realisation (data_seed = 1000 + seed), as every BayesWave run does.

Accuracy = "match" (BayesWave calls it overlap): the normalised inner product between a posterior sample's
reconstructed waveform and the injected waveform, 1 = perfect. Reported: median over the posterior samples
in the second half of the chain so far.

Run:  python3.12 snr_sweep.py            (needs ~/bayeswave from micromamba)
      python3.12 snr_sweep.py --rerun-ours   (our samplers only; keeps the BayesWave runs)
      python3.12 snr_sweep.py --plot-only
"""
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import bayeswave_compare as BC
import tdrjmcmc_qclt as T

SNRS = [10, 20, 40, 80]
SEEDS = 4
BW_NITER, BW_NCHAIN, BW_EVERY = 200_000, 20, 100          # BayesWave writes a chain sample every 100 iterations
BW_CHECKPOINTS = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000]
# (500 and 1000 iterations are only 5 and 10 saved chain samples, so those two points are noisy)
PX_METHODS = ["fisher", "ap_nb", "qclt_nb", "qamr_nb"]
PX_NITER = 40_000
SNR_CAL = 20.0 / 21.380                                    # BayesWave reported 21.38 for prefactor 20
WORK = Path(os.environ.get("SNR_SWEEP_WORK", str(Path.home() / "bayeswave" / "work" / "snr_sweep")))   # on disk:
# /tmp is RAM-backed on this machine and BayesWave's outputs are large
N_WORKERS = 8
OUT = Path(__file__).parent / "figures" / "previous"
DATA = Path(__file__).parent / "data"            # run data and logs (figures/ holds PDFs only)
MDC_CACHE = WORK / "mdc" / "H1_mdc.cache"


# ----------------------------------------------------------------------------- BayesWave
def _chain_lines(d):
    f = d / "chains" / "signal_params_h0.dat.0"
    return sum(1 for _ in open(f)) if f.exists() else 0


def _bw_job(args):
    snr, seed = args
    d = WORK / f"bw_snr{snr}_s{seed}"
    bw_args = BC.bw_args(MDC_CACHE, snr * SNR_CAL, seed, BW_NITER, BW_NCHAIN)
    for old in d.glob("post_*"):                       # leftovers of an interrupted post-processing
        shutil.rmtree(old, ignore_errors=True)
    reused = d.exists() and _chain_lines(d) >= 0.9 * BW_NITER / BW_EVERY
    rc, wall = 0, np.nan
    if not reused:                                     # (re)run BayesWave
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        t0 = time.time()
        rc = BC.bw_run("BayesWave", bw_args, d, d / "bw.log")
        wall = time.time() - t0
    log = (d / "bw.log").read_text()
    m = re.findall(r"Injected network SNR ([\d.]+)", log)
    inj_snr = float(m[-1]) if m else np.nan
    mt = re.findall(r"MCMC for (\d+) iterations took ([\deE.+-]+) s", log)
    sec_per_it = float(mt[-1][1]) / float(mt[-1][0]) if mt else np.nan   # BayesWave's own timing
    avail = _chain_lines(d) * BW_EVERY
    cps = [it for it in BW_CHECKPOINTS if it <= avail]
    if avail not in cps and avail >= BW_CHECKPOINTS[0]:
        cps.append(avail)                              # interrupted run: last available point
    res = dict(snr=snr, seed=seed, rc=rc, wall=wall, sec_per_it=sec_per_it, inj_snr=inj_snr, reused=reused,
               n_done=avail, checkpoints={})
    for it in cps:
        res["checkpoints"][it] = _bw_post_truncated(d, bw_args, it)
    return res


def _bw_post_truncated(d, bw_args, it):
    """BayesWavePost on the first `it` iterations of the chain; returns per-sample overlaps."""
    n_lines = it // BW_EVERY
    c = d / f"post_{it}"
    shutil.rmtree(c, ignore_errors=True)
    c.mkdir()
    for f in d.iterdir():
        if f.name in ("bw.log", "post.log", "post", "chains") or f.name.startswith("post_"):
            continue
        if f.is_dir():
            shutil.copytree(f, c / f.name)
        else:
            os.symlink(f, c / f.name)
    (c / "chains").mkdir()
    for f in (d / "chains").iterdir():
        lines = f.read_text().splitlines(True)[:n_lines]
        (c / "chains" / f.name).write_text("".join(lines))
    rc = BC.bw_run("BayesWavePost", [*bw_args, "--0noise"], c, c / "post.log")
    mom = c / "post" / "signal" / "signal_whitened_moments_H1.dat"
    ov = np.loadtxt(mom, ndmin=2)[:, 7] if rc == 0 and mom.exists() else None
    shutil.rmtree(c, ignore_errors=True)               # post-processing outputs are large: keep only the numbers
    return ov


def _repost_job(r):
    """Add missing checkpoints to an existing BayesWave run (post-processing only, no re-run)."""
    d = WORK / f"bw_snr{r['snr']}_s{r['seed']}"
    bw_args = BC.bw_args(MDC_CACHE, r["snr"] * SNR_CAL, r["seed"], BW_NITER, BW_NCHAIN)
    avail = _chain_lines(d) * BW_EVERY
    for it in BW_CHECKPOINTS:
        if it <= avail and it not in r["checkpoints"]:
            r["checkpoints"][it] = _bw_post_truncated(d, bw_args, it)
    return r


def repost():
    from multiprocessing import Pool
    with open(DATA / "snr_sweep.pkl", "rb") as f:
        data = pickle.load(f)
    with Pool(4) as pool:
        data["bw"] = pool.map(_repost_job, data["bw"], chunksize=1)
    with open(DATA / "snr_sweep.pkl", "wb") as f:
        pickle.dump(data, f)


# ----------------------------------------------------------------------------- our samplers
def _px_job(args):
    method, snr, seed = args
    T.DATASET, T.SNR_INJ, T.BIRTH, T.T0, T.N_MIN_HIST = "chirp", float(snr), "tf", 1000, 10
    T.T_MAX = max(50.0, snr**2 / 2)                         # hottest chain flattens the likelihood
    t0 = time.process_time()
    r = T.run(method, 300 + seed, PX_NITER, PX_NITER // 2, data_seed=1000 + seed, record_waves=False)
    return method, snr, seed, r["match"].astype(np.float32), time.process_time() - t0


def rerun_ours(methods=None):
    """Re-run our samplers `methods` (default all), keeping the BayesWave results and our other samplers'
    runs already in data/snr_sweep.pkl."""
    from multiprocessing import Pool
    methods = methods or PX_METHODS
    with open(DATA / "snr_sweep.pkl", "rb") as f:
        data = pickle.load(f)
    t = time.time()
    with Pool(N_WORKERS) as pool:
        new = pool.map(_px_job, [(m, s, k) for m in methods for s in SNRS for k in range(SEEDS)], chunksize=1)
    data["px"] = [x for x in data["px"] if x[0] not in methods] + new
    data["px_noise"] = "data_seed = 1000 + seed (one noise realisation per run)"
    print(f"TD-RJMCMC: {len(data['px'])} runs in {time.time() - t:.0f}s", flush=True)
    with open(DATA / "snr_sweep.pkl", "wb") as f:
        pickle.dump(data, f)


def run_all():
    from multiprocessing import Pool
    WORK.mkdir(parents=True, exist_ok=True)
    if not MDC_CACHE.exists():
        raise SystemExit(f"missing {MDC_CACHE}: run bayeswave_compare.py make-mdc first")
    ncpu = N_WORKERS
    t = time.time()
    with Pool(ncpu) as pool:
        bw = pool.map(_bw_job, [(s, k) for s in SNRS for k in range(SEEDS)], chunksize=1)
    print(f"BayesWave: {len(bw)} runs in {time.time() - t:.0f}s", flush=True)
    t = time.time()
    with Pool(ncpu) as pool:
        px = pool.map(_px_job, [(m, s, k) for m in PX_METHODS for s in SNRS for k in range(SEEDS)], chunksize=1)
    print(f"TD-RJMCMC: {len(px)} runs in {time.time() - t:.0f}s", flush=True)
    with open(DATA / "snr_sweep.pkl", "wb") as f:
        pickle.dump(dict(bw=bw, px=px), f)


# ----------------------------------------------------------------------------- figure
def summarise(data):
    """Per SNR and sampler: median posterior match after a given effort (iterations, model updates, CPU s)."""
    rows = {}
    rate = {}
    for r in data["bw"]:
        if np.isfinite(r.get("sec_per_it", np.nan)):
            rate.setdefault(r["snr"], []).append(r["sec_per_it"])
    for r in data["bw"]:
        spi = r.get("sec_per_it", np.nan)
        if not np.isfinite(spi):
            spi = np.median(rate.get(r["snr"], [np.nan]))
        for it, ov in r["checkpoints"].items():
            if ov is None or len(ov) == 0:
                continue
            rows.setdefault(("BayesWave", r["snr"]), []).append(
                dict(it=it, updates=it * 100, cpu=spi * it, match=np.median(ov), inj_snr=r["inj_snr"]))
    import bw_ref                                       # BayesWave with settings matched to ours (SNR 20, 40)
    for snr in SNRS:
        for r in bw_ref.runs(snr, "matched"):
            for it, ov in r["checkpoints"].items():
                if ov is None or len(ov) == 0 or it < 500:  # same first checkpoint as the default runs
                    continue
                rows.setdefault(("BayesWave_matched", snr), []).append(
                    dict(it=it, updates=it * 100, cpu=r["sec_per_it"] * it, match=np.median(ov), inj_snr=r["inj_snr"]))
    grid = np.unique(np.geomspace(500, PX_NITER, 12).astype(int))
    for m, snr, seed, match, cpu in data["px"]:
        for it in grid:
            rows.setdefault((m, snr), []).append(
                dict(it=it, updates=it, cpu=cpu * it / PX_NITER, match=float(np.median(match[it // 2:it])),
                     inj_snr=snr))
    return rows


LABEL = {"BayesWave": "BayesWave, default settings (real; classical)",
         "BayesWave_matched": "BayesWave, matched settings (real; classical)", "fisher": "Fisher TD-RJMCMC (ours; classical)", "ap_nb": "AP, Tian et al. (ours; classical)",
         "qclt_nb": "QCLT-AP (ours; quantum kick, simulated)",
         "qamr_nb": "QCLT + AMR (ours; quantum kick, simulated)"}
COLOR = {"BayesWave": "#0b0b0b", "BayesWave_matched": "#9d9c97", "fisher": "#2a78d6", "ap_nb": "#eda100", "qclt_nb": "#e87ba4", "qamr_nb": "#d62f2f"}
MARK = {"BayesWave": "D", "BayesWave_matched": "d", "fisher": "o", "ap_nb": "s", "qclt_nb": "^", "qamr_nb": "*"}


def plot():
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    with open(DATA / "snr_sweep.pkl", "rb") as f:
        data = pickle.load(f)
    rows = summarise(data)
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": T.MUTED, "axes.labelcolor": T.INK, "xtick.color": T.MUTED, "ytick.color": T.MUTED,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    samplers = ["BayesWave", "BayesWave_matched", *[m for m in PX_METHODS if any(x[0] == m for x in data["px"])]]
    fig = plt.figure(figsize=(7.1, 6.3))
    gs = fig.add_gridspec(3, len(SNRS), height_ratios=[1, 1, 1.15], hspace=0.62, wspace=0.28)
    for col, snr in enumerate(SNRS):
        for row, key, xl in ((0, "updates", "model updates per chain"), (1, "cpu", "CPU seconds (one core)")):
            ax = fig.add_subplot(gs[row, col])
            for s in samplers:
                rr = rows.get((s, snr), [])
                if not rr:
                    continue
                by_it = {}
                for r in rr:
                    by_it.setdefault(r["it"], []).append(r)
                its = sorted(by_it)
                x = [np.median([r[key] for r in by_it[i]]) for i in its]
                y = np.array([[r["match"] for r in by_it[i]] for i in its], dtype=object)
                med = [np.median(v) for v in y]
                lo = [np.min(v) for v in y]
                hi = [np.max(v) for v in y]
                ax.fill_between(x, lo, hi, color=COLOR[s], alpha=0.14, lw=0)
                ax.plot(x, med, marker=MARK[s], ms=3, color=COLOR[s], lw=1.1, label=LABEL[s])
            ax.set_xscale("log")
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(xl)
            if col == 0:
                ax.set_ylabel("match with injection")
            ax.set_title(f"SNR {snr}" if row == 0 else "", loc="left")
            ax.grid(color=T.GRIDC, lw=0.6); ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    # bottom row: final accuracy vs SNR
    ax = fig.add_subplot(gs[2, :2])
    for s in samplers:
        xs, med, lo, hi = [], [], [], []
        for snr in SNRS:
            rr = rows.get((s, snr), [])
            if not rr:
                continue
            last = max(r["it"] for r in rr)
            v = [r["match"] for r in rr if r["it"] == last]
            xs.append(snr); med.append(np.median(v)); lo.append(min(v)); hi.append(max(v))
        if not xs:
            continue
        ax.errorbar(np.array(xs) * {"BayesWave": 0.95, "BayesWave_matched": 0.93, "fisher": 0.98, "ap_nb": 1.02, "qclt_nb": 1.05, "qamr_nb": 1.08}[s], med,
                    yerr=[np.array(med) - lo, np.array(hi) - med], marker=MARK[s], ms=4, color=COLOR[s],
                    lw=1.1, capsize=0, label=LABEL[s])
    ax.set_xscale("log"); ax.set_xticks(SNRS); ax.set_xticklabels([str(s) for s in SNRS])
    ax.set_xlabel("signal-to-noise ratio (SNR)")
    ax.set_ylabel("final match")
    ax.set_title("End of run: accuracy vs signal strength", loc="left")
    ax.grid(color=T.GRIDC, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax2 = fig.add_subplot(gs[2, 2:])
    ax2.axis("off")
    h, l = fig.axes[2].get_legend_handles_labels()       # SNR 20 panel: has every sampler
    ax2.legend(h, l, loc="upper left", frameon=False, fontsize=7.5, title="sampler", title_fontsize=7.5,
               bbox_to_anchor=(0.0, 1.05))
    ax2.text(0.0, 0.22, "How to read this figure\n"
             "- match: overlap between the reconstructed and\n  the injected waveform (1 = perfect).\n"
             "- top row: accuracy vs number of model updates\n  (one BayesWave iteration = 100 updates).\n"
             "- middle row: accuracy vs CPU time; BayesWave is\n  compiled C code, ours is Python.\n"
             "- classical: runs on a classical computer. Quantum\n  kick, simulated: QCLT circuit simulated exactly\n"
             "  on a classical computer; rest of sampler classical.\n"
             "- lines: median of 4 runs; shading: min-max.\n"
             "- matched settings: 8 chains, noise spectrum and sky\n  fixed; run at SNR 20 and 40 only.\n"
             "- BayesWave's first two points (500, 1000 iterations)\n  rest on only 5-10 saved samples.",
             fontsize=7, color=T.INK, va="top", transform=ax2.transAxes)
    fig.suptitle("Real BayesWave vs our samplers on the same chirp, at four signal strengths", x=0.01, ha="left",
                 fontsize=9, color=T.INK)
    fig.savefig(OUT / "snr_sweep.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / "snr_sweep.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / "snr_sweep.pdf")
    for s in samplers:
        print(f"{LABEL[s]:28s} " + "  ".join(
            f"SNR {snr}: {np.median([r['match'] for r in rows.get((s, snr), []) if r['it'] == max(x['it'] for x in rows.get((s, snr), [{'it': 0}]))]):.3f}"
            for snr in SNRS if rows.get((s, snr))))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    if "--repost" in sys.argv:
        repost()
    elif "--rerun-ours" in sys.argv:
        rerun_ours()
    elif "--add" in sys.argv:                     # e.g. --add qamr_nb
        rerun_ours(sys.argv[sys.argv.index("--add") + 1].split(","))
    elif "--plot-only" not in sys.argv:
        run_all()
    plot()
