# Quickstart

## Install

```bash
pip install fourierslm
```

In a conda environment, activate it and run the same command. With uv, use `uv add fourierslm`. The only
dependencies are numpy and scipy.

## Units and conventions

- Everything is in SI units, and angles are in radians.
- Fields are complex analytic fields in V/m, so the intensity is $I = \frac{1}{2}\varepsilon_0 c|E|^2$ in W/m²,
  given by `beam.intensity`. Powers are conserved from the SLM to every plane.
- The plane wave is $e^{i(kz - \omega t)}$, so a phase that grows along $+X$ on the SLM moves the focus toward $+x$.
  The constant phase $e^{ikz}$ of the carrier is left out.
- Coordinates are 1D arrays, (X, Y) on the SLM and (x, y) at the focus, and fields have the shape `(len(Y), len(X))`.
  `grid(n, pitch)` gives the pixel coordinates, with the axis on pixel `n // 2`.
- `beam` gives fields, not intensities, and the widths are radii of the intensity at 1/e².
- Fields in complex64 are propagated in complex64, about twice as fast, with errors around 10⁻⁶.

## Example

A Gaussian beam of 1 W on a 1920 × 1200 SLM with 8 µm pixels, a lens of 50 cm, at 1030 nm. The SLM shows a vortex of
charge 1 and a grating that moves the focus by 100 µm along $x$. The focus is a ring with a dark center.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   from fourierslm import beam, grid, phase, propagate

   lam, f, pitch = 1030e-9, 0.5, 8e-6
   X, Y = grid(1920, pitch), grid(1200, pitch)
   pattern = phase.vortex(X, Y, 1) + phase.grating(X, Y, lam, angle=100e-6 / f)
   E = beam.gaussian(X, Y, w=2e-3, power=1.0) * np.exp(1j * pattern)

   E_focus, x, y = propagate.focus(E, pitch, lam, f)
   sx, sy = (x > -50e-6) & (x < 250e-6), np.abs(y) < 150e-6
   I = beam.intensity(E_focus[np.ix_(sy, sx)])
   x, y = x[sx], y[sy]

   fig, ax = plt.subplots(1, 2)
   ax[0].imshow(phase.wrap(pattern), extent=[X[0] * 1e3, X[-1] * 1e3, Y[0] * 1e3, Y[-1] * 1e3], origin="lower",
                cmap="twilight", vmin=0, vmax=2 * np.pi)
   ax[0].set(xlabel="X [mm]", ylabel="Y [mm]", title="phase on the SLM, 0 to 2π")
   im = ax[1].imshow(I * 1e-4, extent=[x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6], origin="lower")
   ax[1].set(xlabel="x [µm]", ylabel="y [µm]", title="intensity at the focus")
   fig.colorbar(im, ax=ax[1], label="W/cm²")
   fig.tight_layout()
```

The [user guide](user_guide/index.md) explains the setup and the propagators.
