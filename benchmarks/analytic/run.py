"""Benchmark fourierslm against the analytic Gaussian beam, and time it.

A Gaussian beam with a flat wavefront on the SLM stays Gaussian through the lens, with the complex radius
q_out = (A q_in + B)/(C q_in + D), so the Collins integral has the closed form

    E(r, z) = E_0 / (A + B/q_in) · exp[ikr²/(2 q_out)],    q_in = −iπw²/λ.

Figures go to docs/_static/benchmark, tables to docs/benchmark_results.md. Run from the root of fourierslm:
    uv run --with matplotlib python benchmarks/analytic/run.py
"""

import os
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.constants import c, epsilon_0

import fourierslm as fs
from fourierslm import beam, propagate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DOCS = HERE.parents[1] / "docs"
FIGS = DOCS / "_static" / "benchmark"
LAM, F, PITCH, W, P = 1030e-9, 0.5, 8e-6, 1.2e-3, 1.0  # 1/e² radius W on the SLM, power P
X, Y = fs.grid(1920, PITCH), fs.grid(1200, PITCH)  # a 1920 × 1200 SLM
W0 = LAM * F / (np.pi * W)
ZR = np.pi * W0**2 / LAM
Z = np.linspace(-4, 4, 81) * ZR
SCAN = "z from −4 z_R to 4 z_R"
L_SLM = (F, 0.3)  # SLM in the front focal plane, and 20 cm closer to the lens


def analytic(x, y, z, L):
    """Gaussian beam through the setup, on the window (x, y), shape (len(z), len(y), len(x))."""
    k, z = 2 * np.pi / LAM, np.atleast_1d(z)[:, None, None]
    A, B, C, D = -z / F, F + z * (1 - L / F), -1 / F, 1 - L / F
    q_in = -1j * np.pi * W**2 / LAM
    q = (A * q_in + B) / (C * q_in + D)
    E0 = np.sqrt(2 * 2 * P / (np.pi * W**2) / (epsilon_0 * c))
    return E0 / (A + B / q_in) * np.exp(1j * k * (x[None, None, :] ** 2 + y[None, :, None] ** 2) / (2 * q))


def err(a, b):
    """Largest difference relative to the largest value of b."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def clock(fun, repeat=5):
    """Best of repeat run times [s], after one warm-up run."""
    fun()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fun()
        times.append(time.perf_counter() - start)
    return min(times)


def main():
    """Accuracy and run time, tables and figure."""
    E = beam.gaussian(X, Y, W, P).astype(complex)
    FIGS.mkdir(parents=True, exist_ok=True)
    lines = accuracy(E) + timing(E)
    (DOCS / "benchmark_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(E)


def accuracy(E):
    """Field against the analytic beam, relative to its peak in each case."""
    x = np.linspace(-3, 3, 256) * W0
    lines = ["| SLM–lens distance | method | precision | planes | error |", "|---|---|---|---|---|"]
    for L in L_SLM:
        ref = analytic(x, x, Z, L)
        for dtype, name in ((complex, "complex128"), (np.complex64, "complex64")):
            num = propagate.focus(E.astype(dtype), X, Y, x, x, LAM, F, Z, slm_distance=L)
            lines.append(f"| {L * 100:.0f} cm | `focus` | {name} | {SCAN} | {err(num, ref):.1e} |")
        u = fs.grid(512, PITCH)
        num = propagate.free(propagate.focus(E, X, Y, u, u, LAM, F, slm_distance=L), PITCH, LAM, Z)
        s = np.abs(u) <= 3 * W0
        e = err(num[:, s][:, :, s], analytic(u[s], u[s], Z, L))
        lines.append(f"| {L * 100:.0f} cm | `focus` on 512², then `free` | complex128 | {SCAN} | {e:.1e} |")
        for z in (0.0, ZR):
            G, xf, yf = propagate.focus_fft(E, PITCH, LAM, F, z, slm_distance=L)
            sx = np.abs(xf) <= 3 * W0
            sy = np.abs(yf) <= 3 * W0
            e = err(G[np.ix_(sy, sx)], analytic(xf[sx], yf[sy], z, L)[0])
            lines.append(f"| {L * 100:.0f} cm | `focus_fft` | complex128 | z = {z / ZR:.0f} z_R | {e:.1e} |")
    caption = (
        f"Gaussian beam of radius {W * 1e3:.1f} mm on a 1920 × 1200 SLM of {PITCH * 1e6:.0f} µm pixels, "
        f"{LAM * 1e9:.0f} nm, f = {F * 100:.0f} cm, so w₀ = {W0 * 1e6:.0f} µm and z_R = {ZR * 100:.1f} cm. "
        "Largest difference over a window of ±3w₀, relative to the peak.\n"
    )
    return ["### Accuracy\n", caption, *lines, ""]


def timing(E):
    """Run times of the propagators on the same SLM."""
    lines = ["| case | complex128 [ms] | complex64 [ms] |", "|---|---|---|"]
    cases = [
        (f"`focus`, window {n}²", lambda E, n=n: propagate.focus(E, X, Y, fs.grid(n, 2e-6), fs.grid(n, 2e-6), LAM, F))
        for n in (128, 256, 512)
    ]
    z = np.linspace(-1e-2, 1e-2, 101)
    w = fs.grid(256, 2e-6)
    cases += [
        ("`focus`, window 256², 101 planes", lambda E: propagate.focus(E, X, Y, w, w, LAM, F, z, slm_distance=0.3)),
        (
            "`focus` on 256², then `free`, 101 planes",
            lambda E: propagate.free(propagate.focus(E, X, Y, w, w, LAM, F, slm_distance=0.3), 2e-6, LAM, z),
        ),
        ("`focus_fft`, pad 2 (3840 × 2400)", lambda E: propagate.focus_fft(E, PITCH, LAM, F, slm_distance=0.3)),
        ("`free`, 1920 × 1200", lambda E: propagate.free(E, PITCH, LAM, 0.1)),
    ]
    fields = [E.astype(complex), E.astype(np.complex64)]
    for name, fun in cases:
        t = [clock(lambda Ed=Ed, fun=fun: fun(Ed), 2 if "planes" in name else 5) for Ed in fields]
        lines.append(f"| {name} | {t[0] * 1e3:.0f} | {t[1] * 1e3:.0f} |")
    cpuinfo = Path("/proc/cpuinfo").read_text() if Path("/proc/cpuinfo").exists() else ""  # Linux only
    models = [line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")]
    cpu = models[0] if models else platform.processor() or platform.machine()
    caption = f"The same 1920 × 1200 SLM. Best of 5 runs (2 for the scan) on {cpu}, {os.cpu_count()} logical cores.\n"
    return ["\n### Run time\n", caption, *lines, ""]


def figure(E):
    """Focal region of the Gaussian beam, x–z map, on-axis intensity and Gouy phase, both SLM–lens distances."""
    x = np.linspace(-4, 4, 161) * W0
    zero = np.zeros(1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    I = beam.intensity(propagate.focus(E, X, Y, x, zero, LAM, F, Z)[:, 0, :])
    ax[0].imshow(I.T, extent=[Z[0] / ZR, Z[-1] / ZR, x[0] / W0, x[-1] / W0], aspect="auto", origin="lower", cmap="bone")
    ax[0].plot(Z / ZR, np.sqrt(1 + (Z / ZR) ** 2), "w--", lw=1, label="analytic w(z)")
    ax[0].plot(Z / ZR, -np.sqrt(1 + (Z / ZR) ** 2), "w--", lw=1)
    ax[0].set(xlabel="z / z_R", ylabel="x / w₀", title="intensity, L = f", ylim=(x[0] / W0, x[-1] / W0))
    ax[0].legend(loc="upper right", fontsize=8)
    for L, color in zip(L_SLM, ("C0", "C1"), strict=True):
        num = propagate.focus(E, X, Y, zero, zero, LAM, F, Z, slm_distance=L)[:, 0, 0]
        ref = analytic(zero, zero, Z, L)[:, 0, 0]
        label = f"L = {L * 100:.0f} cm"
        ax[1].plot(Z / ZR, beam.intensity(num) * 1e-4, ".", color=color, ms=4, label=f"fourierslm, {label}")
        ax[1].plot(Z / ZR, beam.intensity(ref) * 1e-4, "-", color=color, lw=1, label=f"analytic, {label}")
        ax[2].plot(Z / ZR, np.unwrap(np.angle(num)), ".", color=color, ms=4)
        ax[2].plot(Z / ZR, np.unwrap(np.angle(ref)), "-", color=color, lw=1)
    ax[1].set(xlabel="z / z_R", ylabel="on-axis intensity [W/cm²]")
    ax[1].legend(fontsize=7)
    ax[2].set(xlabel="z / z_R", ylabel="on-axis phase [rad]")
    fig.tight_layout()
    fig.savefig(FIGS / "gaussian.png", dpi=150)


if __name__ == "__main__":
    main()
