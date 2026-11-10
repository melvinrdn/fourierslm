"""Benchmark fourierslm against the analytic Gaussian beam.

A Gaussian beam of radius w with a flat wavefront on the SLM gives in the back focal plane, by the FT formula with the
curvature of Goodman (5-19), a Gaussian of radius w₀ = λf/(πw) with the curvature (1 − L/f)/f. Free space adds z to
its complex radius 1/q₀ = (1 − L/f)/f + 2i/(kw₀²), so that

    E(r, z) = E_f · q₀/(q₀ + z) · exp[ikr²/(2(q₀ + z))],    E_f = −i·√I₀,

with E_f the on-axis field in the back focal plane and I₀ = 2P/(πw₀²).

Figures go to docs/_static/benchmark, tables to docs/benchmark_results.md. Run from the root of fourierslm:
    uv run --with matplotlib python benchmarks/analytic/run.py
"""

from pathlib import Path

import matplotlib
import numpy as np

from fourierslm import focus_grid, intensity, propagate, slm_field, slm_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DOCS = HERE.parents[1] / "docs"
FIGS = DOCS / "_static" / "benchmark"
LAM, F, PITCH, W, P = 1030e-9, 0.5, 8e-6, 1.2e-3, 1.0  # 1/e² radius W on the SLM, power P
X, Y = slm_grid(n=1200, pitch=PITCH)  # the short side of a 1920 × 1200 SLM
x, y = focus_grid(X, Y, LAM, F)
W0 = LAM * F / (np.pi * W)
ZR = np.pi * W0**2 / LAM
Z = np.linspace(-4, 4, 81) * ZR
SCAN = "z from −4 z_R to 4 z_R"
L_SLM = (F, 0.3)  # SLM in the front focal plane, and 20 cm closer to the lens
CROP = 2.5e-3  # half width of the focal plane kept for free, 4 radii at 4 z_R


def analytic(x, y, z, L):
    """Gaussian beam through the setup, on the window (x, y), shape (len(z), len(y), len(x))."""
    k, z = 2 * np.pi / LAM, np.atleast_1d(z)[:, None, None]
    q0 = 1 / ((1 - L / F) / F + 2j / (k * W0**2))
    E_f = np.sqrt(2 * P / (np.pi * W0**2)) / 1j
    return E_f * q0 / (q0 + z) * np.exp(1j * k * (x[None, None, :] ** 2 + y[None, :, None] ** 2) / (2 * (q0 + z)))


def err(a, b):
    """Largest difference relative to the largest value of b."""
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


def main():
    """Accuracy table and figure."""
    E = slm_field(intensity.gaussian(X, Y, W, P), 0)
    FIGS.mkdir(parents=True, exist_ok=True)
    lines = accuracy(E)
    (DOCS / "benchmark_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(E)


def scan(E, L):
    """Back focal plane by focus on ±CROP, then free to the planes Z, with its coordinates."""
    u = x[np.abs(x) <= CROP]
    return propagate.free(propagate.focus(E, X, Y, u, u, LAM, F, slm_distance=L), u[1] - u[0], LAM, Z), u


def accuracy(E):
    """Field against the analytic beam on ±3w₀, relative to its peak, in the focal plane and over the scan."""
    lines = ["| SLM–lens distance | method | precision | planes | error |", "|---|---|---|---|---|"]
    for L in L_SLM:
        for dtype, name in ((complex, "complex128"), (np.complex64, "complex64")):
            u = x[np.abs(x) <= 3 * W0]
            e = err(propagate.focus(E.astype(dtype), X, Y, u, u, LAM, F, slm_distance=L), analytic(u, u, 0.0, L)[0])
            lines.append(f"| {L * 100:.0f} cm | `focus` | {name} | back focal plane | {e:.1e} |")
        num, u = scan(E, L)
        s = np.abs(u) <= 3 * W0
        e = err(num[:, s][:, :, s], analytic(u[s], u[s], Z, L))
        method = f"`focus`, then `free` on ±{CROP * 1e3:.1f} mm"
        lines.append(f"| {L * 100:.0f} cm | {method} | complex128 | {SCAN} | {e:.1e} |")
    caption = (
        f"Gaussian beam of radius {W * 1e3:.1f} mm on a 1200 × 1200 grid of {PITCH * 1e6:.0f} µm pixels, "
        f"{LAM * 1e9:.0f} nm, f = {F * 100:.0f} cm, so w₀ = {W0 * 1e6:.0f} µm and z_R = {ZR * 100:.1f} cm. "
        "Largest difference over a window of ±3w₀, relative to the peak.\n"
    )
    return ["### Accuracy\n", caption, *lines, ""]


def figure(E):
    """Focal region of the Gaussian beam, x–z map, on-axis intensity and Gouy phase, both SLM–lens distances."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for L, color in zip(L_SLM, ("C0", "C1"), strict=True):
        num, u = scan(E, L)
        c = len(u) // 2
        if L == F:
            near = np.abs(u) <= 4 * W0
            extent = [Z[0] / ZR, Z[-1] / ZR, u[near][0] / W0, u[near][-1] / W0]
            ax[0].imshow(np.abs(num[:, c, near]).T ** 2, extent=extent, aspect="auto", origin="lower", cmap="bone")
        on_axis, ref = num[:, c, c], analytic(np.zeros(1), np.zeros(1), Z, L)[:, 0, 0]
        label = f"L = {L * 100:.0f} cm"
        ax[1].plot(Z / ZR, np.abs(on_axis) ** 2 * 1e-4, ".", color=color, ms=4, label=f"fourierslm, {label}")
        ax[1].plot(Z / ZR, np.abs(ref) ** 2 * 1e-4, "-", color=color, lw=1, label=f"analytic, {label}")
        ax[2].plot(Z / ZR, np.unwrap(np.angle(on_axis)), ".", color=color, ms=4)
        ax[2].plot(Z / ZR, np.unwrap(np.angle(ref)), "-", color=color, lw=1)
    ax[0].plot(Z / ZR, np.sqrt(1 + (Z / ZR) ** 2), "w--", lw=1, label="analytic w(z)")
    ax[0].plot(Z / ZR, -np.sqrt(1 + (Z / ZR) ** 2), "w--", lw=1)
    ax[0].set(xlabel="z / z_R", ylabel="x / w₀", title="intensity, L = f", ylim=extent[2:])
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].set(xlabel="z / z_R", ylabel="on-axis intensity [W/cm²]")
    ax[1].legend(fontsize=7)
    ax[2].set(xlabel="z / z_R", ylabel="on-axis phase [rad]")
    fig.tight_layout()
    fig.savefig(FIGS / "gaussian.png", dpi=150)


if __name__ == "__main__":
    main()
