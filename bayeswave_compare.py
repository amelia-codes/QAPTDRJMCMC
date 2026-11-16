"""Real BayesWave (LIGO, conda-forge 1.1.3) on the same chirp, across SNRs.

The injection is the chirp of tdrjmcmc_qclt.make_data() (DATASET = "chirp"), coloured by the amplitude
spectral density of BayesWave's simulated Advanced LIGO noise (`--H1-cache LALSimAdLIGO`), so that the
*whitened* waveform BayesWave sees has the same shape as the white-noise injection of our TD-RJMCMC samplers.
It is normalised to optimal SNR 1 against that PSD and written to a frame file; `--MDC-prefactor` then sets
the SNR directly.

Must be run with the BayesWave environment's Python for the frame writing (`make-mdc`); the rest only calls
the BayesWave executables through micromamba.

    MAMBA_ROOT_PREFIX=~/bayeswave/root ~/bayeswave/bin/micromamba run -p ~/bayeswave/env \\
        python bayeswave_compare.py make-mdc <PSD file written by BayesWave> <out dir>
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TRIG = 900000000
SRATE_FRAME = 4096
FRAME_START, FRAME_DUR = TRIG - 16, 288           # covers the PSD stretch [TRIG, TRIG+256] and the segment
CHANNEL = "H1:BW-MDC"
FLOW, FHIGH = 32.0, 512.0                          # analysis band (srate 1024)
MICROMAMBA = Path.home() / "bayeswave" / "bin" / "micromamba"
ENV = Path.home() / "bayeswave" / "env"
ROOT = Path.home() / "bayeswave" / "root"


def chirp_white(t, kind="chirp"):
    """Whitened waveform of tdrjmcmc_qclt.make_data() as a function of time from the segment centre
    (0.5 s there, the trigger time here). kind = "chirp" (short chirp) or "hard" (wide chirp + extra burst)."""
    if kind == "chirp":
        return np.exp(-(t / 0.03) ** 2) * np.cos(2 * np.pi * (120 * t + 0.5 * 3000 * t**2) + 0.7)
    chirp = np.exp(-(t / 0.1) ** 2) * np.cos(2 * np.pi * (200 * t + 0.5 * 700 * t**2) + 0.3)
    tb, tau = -0.28, 5 / (2 * np.pi * 280)                   # burst at 0.22 s of the 1 s segment
    burst = np.exp(-((t - tb) / tau) ** 2) * np.cos(2 * np.pi * 280 * (t - tb) + 1.1)
    return chirp / np.linalg.norm(chirp) * math.sqrt(0.75) + burst / np.linalg.norm(burst) * math.sqrt(0.25)


def make_mdc(psd_file, out_dir, kind="chirp"):
    import lal
    import lalframe

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f_psd, s_psd = np.loadtxt(psd_file, unpack=True)
    # waveform on an 8 s window around the trigger
    n = 8 * SRATE_FRAME
    t = (np.arange(n) - n // 2) / SRATE_FRAME
    w = chirp_white(t, kind)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SRATE_FRAME)
    S = np.interp(f, f_psd, s_psd)
    band = (f >= FLOW) & (f < FHIGH) & (S > 0)
    H = np.where(band, W * np.sqrt(np.where(band, S, 0.0)), 0.0)          # colour by the ASD
    df = f[1] - f[0]
    Hc = H / SRATE_FRAME                                                   # continuous-FT normalisation
    snr = math.sqrt(4 * np.sum(np.abs(Hc[band]) ** 2 / S[band]) * df)
    h = np.fft.irfft(H / snr, n)                                           # optimal SNR = 1
    # embed in the full frame span
    full = np.zeros(FRAME_DUR * SRATE_FRAME)
    i0 = (TRIG - FRAME_START) * SRATE_FRAME - n // 2
    full[i0:i0 + n] = h
    ts = lal.CreateREAL8TimeSeries(CHANNEL, lal.LIGOTimeGPS(FRAME_START), 0.0, 1.0 / SRATE_FRAME,
                                   lal.StrainUnit, len(full))
    ts.data.data[:] = full
    fname = out_dir / f"H-H1_BW_MDC-{FRAME_START}-{FRAME_DUR}.gwf"
    frame = lalframe.FrameNew(lal.LIGOTimeGPS(FRAME_START), FRAME_DUR, "BWMDC", 0, 0, 0)
    lalframe.FrameAddREAL8TimeSeriesProcData(frame, ts)
    lalframe.FrameWrite(frame, str(fname))
    cache = out_dir / "H1_mdc.cache"
    cache.write_text(f"H H1_BW_MDC {FRAME_START} {FRAME_DUR} file://localhost{fname}\n")
    np.save(out_dir / "whitened_template.npy", w)                          # for overlaps later
    print(f"wrote {fname} and {cache}; peak strain {np.abs(h).max():.3e} at SNR 1")
    return cache


def bw_args(cache, snr, seed, niter, nchain, extra=()):
    return [
        "--ifo", "H1", "--H1-flow", str(int(FLOW)), "--H1-cache", "LALSimAdLIGO", "--H1-channel", "LALSimAdLIGO",
        "--trigtime", f"{TRIG}.00", "--srate", "1024", "--seglen", "4", "--psdstart", str(TRIG),
        "--psdlength", "256", "--dataseed", str(1000 + seed), "--chainseed", str(2000 + seed),
        "--MDC-cache", f"[{cache}]", "--MDC-channel", f"[{CHANNEL}]", "--MDC-prefactor", f"{snr}",
        "--signalOnly", "--noClean", "--Niter", str(niter), "--Nchain", str(nchain), *extra,
    ]


def bw_run(exe, args, cwd, log):
    """Run a BayesWave executable from the environment directly (parallel `micromamba run` calls collide on
    micromamba's lock file)."""
    with open(log, "w") as fh:
        return subprocess.run([str(ENV / "bin" / exe), *args], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


if __name__ == "__main__":
    if sys.argv[1] == "make-mdc":
        make_mdc(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "chirp")
