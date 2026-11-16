"""Long runs of the three wavelet TD-RJMCMC samplers (see tdrjmcmc_qclt.py) and the extra figures:

  figures/previous/tdrjmcmc_accuracy.pdf      accuracy vs iteration: error of the running posterior-mean
                                           waveform, sample mismatch 1 - match, and the Tian et al.
                                           two-run statistics R_c1 / R_c2 (log-log, median and IQR)
  figures/previous/tdrjmcmc_trajectories.pdf  where the cold chain's wavelets go: (t0, f0) trajectory over
                                           the data's time-frequency map, and the posterior waveform
  figures/previous/tdrjmcmc_burnin.pdf        every parameter of every cold-chain wavelet over the run

Run:   python3.12 tdrjmcmc_figures.py              # long runs + figures
       python3.12 tdrjmcmc_figures.py --plot-only  # figures from the saved runs
       python3.12 tdrjmcmc_figures.py --add qamr   # add QCLT + AMR runs to the store, then redraw
"""
import itertools
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import tdrjmcmc_qclt as T

N_SEEDS = 8
N_ITER = 100_000
N_ADAPT = 20_000
PARAM_EVERY = 20
MON = 250                                   # monitoring grid (iterations)
OUT = Path(__file__).parent / "figures" / "previous"
STORE = Path(os.environ.get("TDRJ_STORE", Path.home() / "bayeswave" / "work" / "tdrjmcmc_long_runs.pkl"))  # ~260 MB


def _job(args):
    m, s = args
    r = T.run(m, 1000 + s, N_ITER, N_ADAPT, record_params_every=PARAM_EVERY)
    return m, s, r


def run_all(methods=None):
    """Run the long runs for `methods` (default: all) and merge them into STORE (other methods are kept)."""
    from multiprocessing import Pool
    methods = methods or T.METHODS
    jobs = [(m, s) for m in methods for s in range(N_SEEDS)]
    t0 = time.time()
    with Pool(min(len(jobs), int(os.environ.get("N_WORKERS", os.cpu_count() or 4)))) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    print(f"{len(jobs)} runs x {N_ITER} iterations in {time.time() - t0:.0f}s")
    keep = load() if STORE.exists() else {}
    for m in methods:
        keep[m] = []
    for m, s, r in res:
        keep.setdefault(m, []).append({k: r[k] for k in ("logL", "K", "match", "waves", "params", "cpu", "acc")})
    with open(STORE, "wb") as f:
        pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)


def load():
    with open(STORE, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- accuracy statistics
def running_posterior_error(waves, h_ref):
    """||mean(waves[t/2:t]) - h_ref|| / ||h_ref|| on the monitoring grid (second-half window)."""
    cs = np.concatenate([np.zeros((1, waves.shape[1])), np.cumsum(waves, axis=0, dtype=np.float64)])
    steps = np.arange(MON // T.WAVE_EVERY, len(waves) + 1, MON // T.WAVE_EVERY)
    err = np.empty(len(steps))
    for k, s in enumerate(steps):
        a = s // 2
        mean = (cs[s] - cs[a]) / (s - a)
        err[k] = np.linalg.norm(mean - h_ref) / np.linalg.norm(h_ref)
    return steps * T.WAVE_EVERY, err


def rc_curves(waves_list):
    """Median over run pairs of Tian et al.'s R_c1 and R_c2 on the monitoring grid."""
    W = np.stack(waves_list)
    cs = np.concatenate([np.zeros((W.shape[0], 1, W.shape[2])), np.cumsum(W, 1, dtype=np.float64)], 1)
    cs2 = np.concatenate([np.zeros((W.shape[0], 1, W.shape[2])), np.cumsum(W.astype(np.float64) ** 2, 1)], 1)
    steps = np.arange(MON // T.WAVE_EVERY, W.shape[1] + 1, MON // T.WAVE_EVERY)
    r1, r2 = [], []
    for s in steps:
        a = s // 2
        mu = (cs[:, s] - cs[:, a]) / (s - a)
        sd = np.sqrt(np.maximum((cs2[:, s] - cs2[:, a]) / (s - a) - mu**2, 0))
        p1, p2 = [], []
        for i, j in itertools.combinations(range(W.shape[0]), 2):
            den = np.maximum((sd[i] + sd[j]) / 2, 1e-3)
            p1.append(np.mean(np.abs(mu[i] - mu[j]) / den))
            p2.append(np.mean(np.abs(sd[i] - sd[j]) / den))
        r1.append(np.percentile(p1, [25, 50, 75]))
        r2.append(np.percentile(p2, [25, 50, 75]))
    return steps * T.WAVE_EVERY, np.array(r1), np.array(r2)


# ----------------------------------------------------------------------------- style
INK, MUTED, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def finish(ax):
    ax.grid(color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    if os.environ.get("FIG_PREVIEW_DIR"):           # PNG previews only outside the repo (PDF is the figure)
        fig.savefig(Path(os.environ["FIG_PREVIEW_DIR"]) / f"{name}.png", dpi=110, bbox_inches="tight")
    print("saved", OUT / f"{name}.pdf")


# ----------------------------------------------------------------------------- figure: accuracy
def _logspaced(n, k=900):
    return np.unique(np.geomspace(1, n, k).astype(int)) - 1


def fig_accuracy(runs):
    """(a) error vs the injection (noise-limited), (b) sampling error vs a leave-one-out posterior
    reference (what the sampler controls), (c) 1 - match of the current sample, (d) Tian et al. R_c."""
    from matplotlib import pyplot as plt
    style()
    d, h_inj = T.make_data()
    h_ref = h_inj[T.W0:T.W1]
    norm = np.linalg.norm(h_ref)
    # posterior mean of every run over its last half, for leave-one-out references
    all_means = {(m, i): r["waves"][len(r["waves"]) // 2:].mean(0) for m in T.METHODS for i, r in enumerate(runs[m])}
    pooled = np.mean(list(all_means.values()), axis=0)
    n_all = len(all_means)
    floor = np.linalg.norm(pooled - h_ref) / norm
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
    axes = axes.ravel()
    summary = {}
    for m in T.METHODS:
        e_inj, e_loo = [], []
        for i, r in enumerate(runs[m]):
            ref_loo = (pooled * n_all - all_means[(m, i)]) / (n_all - 1)      # exclude this run
            cs = np.concatenate([np.zeros((1, r["waves"].shape[1])), np.cumsum(r["waves"], 0, dtype=np.float64)])
            steps = np.arange(MON // T.WAVE_EVERY, len(r["waves"]) + 1, MON // T.WAVE_EVERY)
            mean = np.array([(cs[s_] - cs[s_ // 2]) / (s_ - s_ // 2) for s_ in steps])
            e_inj.append(np.linalg.norm(mean - h_ref, axis=1) / norm)
            e_loo.append(np.linalg.norm(mean - ref_loo, axis=1) / norm)
        it = steps * T.WAVE_EVERY
        for ax, e in ((axes[0], np.array(e_inj)), (axes[1], np.array(e_loo))):
            lo, med, hi = np.percentile(e, [25, 50, 75], axis=0)
            ax.fill_between(it, lo, hi, color=T.COLORS[m], alpha=0.18, lw=0)
            ax.plot(it, med, T.LSTYLE[m], color=T.COLORS[m], lw=1.3, label=T.LABELS[m])
        e_loo = np.array(e_loo)
        summary[m] = dict(final_err=np.median(np.array(e_inj)[:, -1]), err_iqr=tuple(np.percentile(np.array(e_inj)[:, -1], [25, 75])),
                          final_loo=np.median(e_loo[:, -1]), loo_iqr=tuple(np.percentile(e_loo[:, -1], [25, 75])))

        mm = np.array([1 - r["match"] for r in runs[m]])
        ker = np.ones(200) / 200
        mm = np.array([np.convolve(x, ker, mode="valid") for x in mm])
        idx = _logspaced(mm.shape[1])
        x = idx + 200
        lo, med, hi = np.percentile(mm[:, idx], [25, 50, 75], axis=0)
        axes[2].fill_between(x, lo, hi, color=T.COLORS[m], alpha=0.18, lw=0)
        axes[2].plot(x, med, T.LSTYLE[m], color=T.COLORS[m], lw=1.1)
    import bw_ref
    for variant, col, mfc in (("default", "#0b0b0b", None), ("matched", "#9d9c97", "white")):
        bmc = bw_ref.median_curve(20, variant)
        if bmc is not None:
            axes[2].plot(bmc["updates"], 1 - bmc["match"], "D-", ms=3, lw=1.1, color=col, mfc=mfc,
                         label=bw_ref.LABELS[variant])
            axes[2].set_xlim(200, bmc["updates"].max() * 1.5)
    if axes[2].get_legend_handles_labels()[0]:
        axes[2].legend(loc="upper right", frameon=False, fontsize=6.5)
    for m in T.METHODS:
        pass

        it2, r1, r2 = rc_curves([r["waves"] for r in runs[m]])
        axes[3].fill_between(it2, r1[:, 0], r1[:, 2], color=T.COLORS[m], alpha=0.18, lw=0)
        axes[3].plot(it2, r1[:, 1], T.LSTYLE[m], color=T.COLORS[m], lw=1.3)
        axes[3].plot(it2, r2[:, 1], ":", color=T.COLORS[m], lw=1.0)
        below = np.nonzero((r1[:, 1] < 0.2) & (r2[:, 1] < 0.2))[0]
        summary[m]["median_pair_conv"] = it2[below[0]] if len(below) else np.nan
        summary[m]["final_rc1"], summary[m]["final_rc2"] = r1[-1, 1], r2[-1, 1]

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(N_ADAPT, color=MUTED, lw=0.6, ls=":")
        ax.set_xlabel("iteration")
        finish(ax)
    axes[0].axhline(floor, color=INK, lw=0.7)
    axes[0].text(N_ITER, floor * 1.04, f"noise floor {floor:.2f}", fontsize=6.5, color=INK, va="bottom", ha="right")
    axes[0].set_ylabel(r"$\|\bar h - h_\mathrm{inj}\| / \|h_\mathrm{inj}\|$")
    axes[0].set_title("(a) Posterior mean vs injection (ours)", loc="left")
    guide_x = np.array([3e3, N_ITER])
    g0 = np.median([summary[m]["final_loo"] for m in T.METHODS]) * np.sqrt(N_ITER / guide_x[0])
    axes[1].plot(guide_x, g0 * np.sqrt(guide_x[0] / guide_x), color=MUTED, lw=0.8, ls="--")
    axes[1].text(guide_x[0] * 1.3, g0 * 1.05, r"$\propto t^{-1/2}$", color=MUTED, fontsize=7)
    axes[1].set_ylabel(r"$\|\bar h - \bar h_\mathrm{ref}\| / \|h_\mathrm{inj}\|$")
    axes[1].set_title("(b) Sampling error vs leave-one-out posterior (ours)", loc="left")
    axes[2].set_ylabel(r"$1 - \mathrm{match}$ (200-it. average)")
    axes[2].set_title("(c) Current-sample mismatch, incl. BayesWave", loc="left")
    axes[2].text(0.99, 0.03, "ours: one noise realisation (a favourable one);\nBayesWave: one per run. With one per run,\n"
                 "ours also end near 0.08 (snr_sweep.pdf)", transform=axes[2].transAxes, fontsize=6.0, color=T.MUTED,
                 ha="right", va="bottom")
    for k in (0, 1, 3):                                # these need per-sample waveforms
        axes[k].text(0.99, 0.99, "BayesWave: n/a (saves no per-sample waveforms)", transform=axes[k].transAxes,
                     ha="right", va="top", fontsize=5.8, color=T.MUTED)
    axes[2].set_xlabel("model updates per chain")
    axes[3].axhline(0.2, color=INK, lw=0.7)
    axes[3].text(300, 0.205, "Tian et al. threshold 0.2", ha="left", va="bottom", fontsize=6.5, color=INK)
    axes[3].set_ylabel(r"$R_{c1}$ (solid), $R_{c2}$ (dotted)")
    axes[3].set_title("(d) Two-run agreement, median over pairs", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.2, w_pad=1.2)
    save(fig, "tdrjmcmc_accuracy")
    return summary


# ----------------------------------------------------------------------------- figure: trajectories
def tf_map(d):
    """Quadrature matched-filter power of the data on a (t, f) grid (Q = 8), for the background."""
    ts = np.linspace(0.38, 0.62, 97)
    fs = np.linspace(30, 320, 117)
    P = np.zeros((len(fs), len(ts)))
    for j, t0 in enumerate(ts):
        for i, f0 in enumerate(fs):
            th = np.array([t0, f0, 8.0, 0.0, 0.0])
            i0, i1 = T.wave_window(th)
            c = T.wave_on(th, i0, i1)
            th[4] = np.pi / 2
            s = T.wave_on(th, i0, i1)
            P[i, j] = (d[i0:i1] @ c) ** 2 / (c @ c) + (d[i0:i1] @ s) ** 2 / (s @ s)
    return ts, fs, P


def fig_trajectories(runs, seed_idx=0):
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    style()
    d, h_inj = T.make_data()
    ts, fs, P = tf_map(d)
    gray = LinearSegmentedColormap.from_list("g", [SURF, "#d8d7d2", "#9d9c97"])
    seq = LinearSegmentedColormap.from_list(
        "b", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    import bw_ref
    bwd = bw_ref.details()
    ncol = len(T.METHODS) + (bwd is not None)
    fig, axes = plt.subplots(2, ncol, figsize=(7.6, 4.4), gridspec_kw=dict(height_ratios=[1.35, 1]))
    tt = T.TT[T.W0:T.W1]
    unorm = LogNorm(vmin=PARAM_EVERY * 5, vmax=2e7)          # colour = model updates per chain
    for col, m in enumerate(T.METHODS):
        r = runs[m][seed_idx]
        ax = axes[0, col]
        ax.pcolormesh(ts, fs, P, cmap=gray, shading="auto", rasterized=True)
        x = np.linspace(0.44, 0.56, 50)
        ax.plot(x, 120 + 3000 * (x - 0.5), color=INK, lw=0.8, ls="--")
        its, pts = [], []
        for it, arr in r["params"]:
            if len(arr):
                its.append(np.full(len(arr), it)); pts.append(arr)
        its = np.concatenate(its); pts = np.concatenate(pts)
        keep = pts[:, 5] > 1.0                                      # wavelets with SNR > 1
        sc = ax.scatter(pts[keep, 0], pts[keep, 1], c=its[keep], s=np.clip(pts[keep, 5], 1, 12) * 0.6,
                        cmap=seq, norm=unorm, lw=0, alpha=0.6, rasterized=True)
        ax.set_xlim(ts[0], ts[-1]); ax.set_ylim(fs[0], fs[-1])
        ax.set_title(f"({'abcd'[col]}) {T.LABELS[m]}", loc="left", fontsize=7.8)
        ax.set_xlabel(r"$t_0$ (s)")
        if col == 0:
            ax.set_ylabel(r"$f_0$ (Hz)")
            pass

        ax = axes[1, col]
        half = len(r["waves"]) // 2
        post = r["waves"][half:]
        lo, med, hi = np.percentile(post, [5, 50, 95], axis=0)
        ax.fill_between(tt, lo, hi, color=T.COLORS[m], alpha=0.35, lw=0, label="90% credible")
        ax.plot(tt, h_inj[T.W0:T.W1], color=INK, lw=0.8, label="injected")
        ax.plot(tt, med, color=T.COLORS[m], lw=0.9, label="posterior median")
        ax.set_xlim(0.42, 0.58)
        ax.set_xlabel("time (s)")
        if col == 0:
            ax.set_ylabel("whitened strain")

        finish(ax)
        ax.set_title(f"({'efgh'[col]}) Reconstruction", loc="left", fontsize=7.8)
    if bwd is not None:                                       # real BayesWave, SNR-20 run (seed 0)
        col = ncol - 1
        ax = axes[0, col]
        ax.pcolormesh(ts, fs, P, cmap=gray, shading="auto", rasterized=True)
        x = np.linspace(0.44, 0.56, 50)
        ax.plot(x, 120 + 3000 * (x - 0.5), color=INK, lw=0.8, ls="--")
        upd = np.concatenate([np.full(len(w), it * bw_ref.UPDATES_PER_IT) for it, w in zip(bwd["iters"], bwd["params"])])
        pts = np.concatenate([w for w in bwd["params"] if len(w)])
        ax.scatter(pts[:, 0] - 1.5, pts[:, 1], c=np.maximum(upd, 1), s=2.5, cmap=seq, norm=unorm, lw=0, alpha=0.6,
                   rasterized=True)
        ax.set_xlim(ts[0], ts[-1]); ax.set_ylim(fs[0], fs[-1])
        ax.set_title(f"(d) {bw_ref.LABEL}", loc="left", fontsize=7.8)
        ax.set_xlabel(r"$t_0$ (s)")
        ax = axes[1, col]
        h_ref = h_inj[T.W0:T.W1]
        tb = np.round(bwd["time"] * T.FS).astype(int)
        sel = (tb >= T.W0) & (tb < T.W1)
        scale = np.linalg.norm(h_ref) / np.linalg.norm(bwd["injected"][sel])    # same SNR scale as ours
        ax.fill_between(bwd["time"][sel], bwd["lo90"][sel] * scale, bwd["hi90"][sel] * scale, color="#9d9c97",
                        alpha=0.45, lw=0)
        ax.plot(bwd["time"][sel], bwd["injected"][sel] * scale, color=INK, lw=0.8)
        ax.plot(bwd["time"][sel], bwd["med"][sel] * scale, color="#e34948", lw=0.9)
        ax.set_xlim(0.42, 0.58)
        ax.set_xlabel("time (s)")
        finish(ax)
        ax.set_title("(h) Reconstruction (red: BayesWave)", loc="left", fontsize=7.8)
    cax = fig.add_axes([0.915, 0.53, 0.012, 0.33])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("model updates per chain", color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    fig.suptitle("Where the wavelets go, SNR-20 chirp, one run per sampler. Top: wavelet centres in the time-frequency "
                 "plane (grey: data power;\ndashed: injected chirp; marker size ~ wavelet SNR for ours). Bottom: "
                 "posterior median and 90% band vs injected signal (black). BayesWave sees\nLIGO-coloured noise; its "
                 "reconstruction is shown whitened and rescaled to the same SNR.", x=0.01, ha="left", fontsize=7.6,
                 color=INK)
    fig.subplots_adjust(left=0.07, right=0.9, top=0.84, bottom=0.08, hspace=0.45, wspace=0.3)
    save(fig, "tdrjmcmc_trajectories")


def fig_burnin(runs, seed_idx=0, methods=None, n_iter=None, n_adapt=None, param_every=None,
               name="tdrjmcmc_burnin", t0=None, chirp_marks=True, with_bw=False):
    """The trending phase: every parameter of every cold-chain wavelet (no selection), log iteration axis."""
    from matplotlib import pyplot as plt
    methods = methods or T.METHODS
    n_iter, n_adapt = n_iter or N_ITER, n_adapt or N_ADAPT
    param_every = param_every or PARAM_EVERY
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    style()
    seq = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    norm = Normalize(0, 15)
    # (column in the recorded array, label, transform, y-limits, log-y)
    prow = [(0, r"$t_0$ (s)", None, (0.1, 0.9), False),
            (1, r"$f_0$ (Hz)", None, (30, 400), False),
            (2, r"$Q$", None, (2, 30), False),
            (3, r"$A$", np.exp, (0.1, 20), True),
            (4, r"$\phi$ (rad)", None, (0, 2 * np.pi), False),
            (5, "wavelet SNR", None, (0, 25), False)]
    n_rows = len(prow) + 2
    bwd = None
    if with_bw:
        import bw_ref
        bwd = bw_ref.details()
        bw_curve = bw_ref.curves(20)
    ncol = len(methods) + (bwd is not None)
    fig, axes = plt.subplots(n_rows, ncol, figsize=(7.6, 8.8), sharex=True, sharey="row",
                             gridspec_kw=dict(height_ratios=[1] * len(prow) + [0.7, 0.7]))
    for col, m in enumerate(methods):
        r = runs[m][seed_idx]
        its = np.concatenate([np.full(len(a), it) for it, a in r["params"] if len(a)])
        P = np.concatenate([a for it, a in r["params"] if len(a)])
        its = np.maximum(its, param_every / 2)
        for row, (c, lab, tf, lim, logy) in enumerate(prow):
            y = tf(P[:, c]) if tf else P[:, c]
            sc = axes[row, col].scatter(its, y, c=P[:, 5], cmap=seq, norm=norm, s=2.5, lw=0, rasterized=True)
        x = np.arange(1, len(r['K']) + 1)
        axes[-2, col].plot(x, r["K"], color=T.COLORS[m], lw=0.6, drawstyle="steps-post", rasterized=True)
        axes[-1, col].plot(x, r["match"], color=T.COLORS[m], lw=0.6, rasterized=True)
        hit = int(np.argmax(r["match"] >= 0.9)) + 1
        for ax in axes[:, col]:
            ax.axvline(hit, color=INK, lw=0.6, ls="--")
            ax.axvline(n_adapt, color=MUTED, lw=0.6, ls=":")
            if t0:
                ax.axvline(t0, color=MUTED, lw=0.5, ls="-.")
            finish(ax)
        axes[-1, col].text(hit * 1.15, 0.08, f"0.9 at {hit}", fontsize=6.5, color=INK)
        axes[0, col].set_title(f"({'abcd'[col]}) {T.LABELS[m]}", loc="left", fontsize=7.8)
        axes[-1, col].set_xlabel("model updates per chain")
    if bwd is not None:                                     # real BayesWave, SNR-20 run (seed 0)
        col = ncol - 1
        upd = np.concatenate([np.full(len(w), max(it, 1) * bw_ref.UPDATES_PER_IT) for it, w in zip(bwd["iters"], bwd["params"])])
        Pb = np.concatenate([w for w in bwd["params"] if len(w)])
        Pb = Pb.copy(); Pb[:, 0] -= 1.5                      # BayesWave segment time -> our 1 s segment
        for row, (c, lab, tf, lim, logy) in enumerate(prow):
            ax = axes[row, col]
            if c in (3, 5):                                  # amplitude / SNR: not comparable
                ax.text(0.5, 0.5, "not comparable:\nBayesWave amplitudes are in\nstrain units and depend on\nsky position",
                        transform=ax.transAxes, ha="center", va="center", fontsize=6, color=MUTED)
            else:
                ax.scatter(upd, Pb[:, c], s=2.0, lw=0, color=bw_ref.COLOR, alpha=0.5, rasterized=True)
        axes[-2, col].plot(np.maximum(bwd["iters"], 1) * bw_ref.UPDATES_PER_IT, bwd["D"], color=bw_ref.COLOR, lw=0.6,
                           drawstyle="steps-post", rasterized=True)
        c0 = bw_curve[0] if bw_curve else None
        if c0 is not None:
            axes[-1, col].plot(c0["updates"], c0["match"], "D-", ms=2.5, lw=0.8, color=bw_ref.COLOR)
        for ax in axes[:, col]:
            finish(ax)
        axes[0, col].set_title(f"(d) {bw_ref.LABEL}", loc="left", fontsize=7.8)
        axes[-1, col].set_xlabel("model updates per chain")
    for row, (c, lab, tf, lim, logy) in enumerate(prow):
        axes[row, 0].set_ylabel(lab)
        axes[row, 0].set_ylim(*lim)
        if logy:
            axes[row, 0].set_yscale("log")
    if chirp_marks:
        for ax in axes[1]:
            for f in (100, 150):
                ax.axhline(f, color=MUTED, lw=0.4)
        for ax in axes[0]:
            ax.axhspan(0.47, 0.53, color=MUTED, alpha=0.12, lw=0)
    axes[-2, 0].set_ylabel(r"$N$ (wavelets)")
    axes[-1, 0].set_ylabel("match")
    axes[-1, 0].set_ylim(0, 1)
    axes[-1, 0].set_xscale("log")
    axes[-1, 0].set_xlim(10, 2.5e7 if bwd is not None else n_iter)
    cax = fig.add_axes([0.92, 0.55, 0.012, 0.3])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("wavelet SNR", color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=7, length=0)
    fig.suptitle("Burn-in of one run per sampler: all parameters of every cold-chain wavelet (dots; ours coloured by "
                 "wavelet SNR), number of wavelets N, and match.\nOne of our iterations = 1 model update per chain; one "
                 "BayesWave iteration = 100. Dashed: match first reaches 0.9; dash-dot: adaptive move starts;\ndotted: "
                 "adaptation frozen" + ("; shaded t0 band / grey f0 lines: where the chirp is." if chirp_marks else "."),
                 x=0.01, ha="left", fontsize=7.5, color=INK)
    fig.subplots_adjust(left=0.08, right=0.9, top=0.91, bottom=0.05, hspace=0.18, wspace=0.12)
    save(fig, name)


def report(runs, summary):
    print(f"\n{'sampler':18s} {'err vs inj':>11s} {'sampling err (LOO)':>19s} {'IQR':>17s} "
          f"{'median-pair conv':>17s} {'R_c1 final':>11s} {'R_c2 final':>11s} {'ms/iter':>8s}")
    for m in T.METHODS:
        s = summary[m]
        cpu = np.mean([r["cpu"] for r in runs[m]]) / N_ITER * 1e3
        print(f"{T.LABELS[m]:18s} {s['final_err']:11.4f} {s['final_loo']:19.4f} "
              f"{s['loo_iqr'][0]:8.4f}-{s['loo_iqr'][1]:<8.4f} {s['median_pair_conv']:17.0f} "
              f"{s['final_rc1']:11.3f} {s['final_rc2']:11.3f} {cpu:8.2f}")


if __name__ == "__main__":
    if "--add" in sys.argv:                       # e.g. --add qamr: run one more sampler, keep the stored ones
        run_all(sys.argv[sys.argv.index("--add") + 1].split(","))
    elif "--plot-only" not in sys.argv:
        run_all()
    runs = load()
    T.METHODS[:] = [m for m in ("fisher", "ap", "qclt", "qamr") if m in runs]
    summary = fig_accuracy(runs)
    fig_trajectories(runs)
    fig_burnin(runs, t0=T.T0, with_bw=True)
    report(runs, summary)
