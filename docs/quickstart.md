# Quickstart

## Install

```bash
pip install fourierslm
```

In a conda environment, activate it and run the same command. With uv, use `uv add fourierslm`. The only
dependencies are numpy and scipy.

## Units and conventions

- Everything is in SI units, and angles are in radians.
- Fields are in √(W/m²). `slm_field` makes $E = \sqrt{I}\,e^{i\phi}$ from the intensity and the phase, so $|E|^2$ is
  the intensity in W/m² on the SLM and in every plane after it. Powers are conserved from the SLM to every plane.
- For a pulse, give the peak power to the `intensity` functions, `peak_power(energy, duration)` for a Gaussian pulse,
  and $|E|^2$ is the peak intensity.
- The plane wave is $e^{i(kz - \omega t)}$, so a phase that grows along $+X$ on the SLM moves the focus toward $+x$.
  The constant phase $e^{ikz}$ of the carrier is left out.
- Grids are square. `slm_grid` gives the coordinates (X, Y) on the SLM, `focus_grid` the coordinates (x, y) at the
  focus, both 1D with the axis on index `n // 2`, and fields have the shape `(n, n)`.
- The widths of the `intensity` functions are radii of the intensity at 1/e².
- Fields in complex64 are propagated in complex64, about twice as fast, with errors around 10⁻⁶.

## Choosing the grid

`slm_grid(size, n=2048)` takes the side of the square and the number of points. The size must hold the beam, at least
4 beam radii for a Gaussian. Then each one sets one thing at the focus, with the default padding of `focus_grid`.

- With the size fixed, the number of points sets the field of view at the focus, $n\lambda f/(2\,\mathrm{size})$. A
  bigger field of view needs more points.
- With n fixed, the size sets the detail at the focus, $\lambda f/(2\,\mathrm{size})$. Finer detail needs a bigger
  size, and 20 times the beam radius gives about $0.08\,w_0$.

Give `pitch` instead of `size` only when the SLM pixels matter, as for stochastic holograms or quantization. Then the
size is n times the pitch, `slm_grid(n=2048, pitch=8e-6)`.

## Example

A Gaussian beam of 1 W on a square of 20 mm, a lens of 50 cm, at 1030 nm. The SLM shows a vortex of charge 1 and a
grating that moves the focus by 100 µm along $x$. The focus is a ring with a dark center.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   import fourierslm as fs

   lam, f, w = 1030e-9, 0.5, 2e-3
   X, Y = fs.slm_grid(size=10 * w)
   x, y = fs.focus_grid(X, Y, lam, f)

   intensity = fs.intensity.gaussian(X, Y, w, power=1.0)
   phase = fs.phase.vortex(X, Y, 1) + fs.phase.grating(X, Y, lam, angle=100e-6 / f)
   E_slm = fs.slm_field(intensity, phase)
   E_focus = fs.propagate.focus(E_slm, X, Y, x, y, lam, f)

   fig, ax = plt.subplots(1, 2)
   ax[0].imshow(fs.phase.wrap(phase), extent=[X[0] * 1e3, X[-1] * 1e3, Y[0] * 1e3, Y[-1] * 1e3], origin="lower",
                cmap="twilight", vmin=0, vmax=2 * np.pi)
   ax[0].set(xlabel="X [mm]", ylabel="Y [mm]", title="phase on the SLM, 0 to 2π")
   im = ax[1].imshow(np.abs(E_focus) ** 2 * 1e-4, extent=[x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6],
                     origin="lower")
   ax[1].set(xlabel="x [µm]", ylabel="y [µm]", title="intensity at the focus", xlim=(-50, 250), ylim=(-150, 150))
   fig.colorbar(im, ax=ax[1], label="W/cm²")
   fig.tight_layout()
```

The [user guide](user_guide/index.md) explains the setup and the propagators.
