# fourierslm

[![CI](https://github.com/melvinrdn/fourierslm/actions/workflows/ci.yml/badge.svg)](https://github.com/melvinrdn/fourierslm/actions/workflows/ci.yml)
[![Docs](https://github.com/melvinrdn/fourierslm/actions/workflows/docs.yml/badge.svg)](https://melvinrdn.github.io/fourierslm)

Fourier optics of a phase-only SLM: the beam on the SLM, its phase pattern, and the field it makes in the focal region
of a lens, in physical units.

**Documentation:** https://melvinrdn.github.io/fourierslm

## Install

```bash
pip install fourierslm    # or: uv add fourierslm
```

## Example

A Gaussian beam of 1 W on a 1920 × 1200 SLM with 8 µm pixels, a vortex of charge 1, and a lens of 50 cm placed 30 cm
after the SLM. `focus` gives the field in the back focal plane, the Fourier transform of the field on the SLM. `free`
then propagates the central part of it through the focus.

```python
import matplotlib.pyplot as plt
import numpy as np

from fourierslm import beam, grid, phase, propagate

lam, f, pitch = 1030e-9, 0.5, 8e-6
X, Y = grid(1920, pitch), grid(1200, pitch)
E = beam.gaussian(X, Y, w=2e-3, power=1.0)  # field [V/m], w is the 1/e² radius of the intensity
E = E * np.exp(1j * phase.vortex(X, Y, charge=1))
E_focus, x, y = propagate.focus(E, pitch, lam, f, slm_distance=0.3)  # back focal plane

keep = np.abs(x) < 1.5e-3  # ±1.5 mm around the axis, x and y share the grid
E_focus, x = E_focus[np.ix_(keep, keep)], x[keep]
z = np.linspace(-60e-3, 60e-3, 121)
I = beam.intensity(propagate.free(E_focus, x[1] - x[0], lam, z))  # [W/m²], shape (len(z), len(x), len(x))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
c = len(x) // 2
im = ax[0].imshow(I[60] * 1e-4, extent=[x[0] * 1e6, x[-1] * 1e6] * 2, origin="lower", cmap="bone")
ax[0].set(xlabel="x [µm]", ylabel="y [µm]", title="intensity at the focus", xlim=(-250, 250), ylim=(-250, 250))
fig.colorbar(im, ax=ax[0], label="W/cm²")
ax[1].imshow(
    I[:, c, :].T * 1e-4,
    extent=[z[0] * 1e3, z[-1] * 1e3, x[0] * 1e6, x[-1] * 1e6],
    origin="lower",
    aspect="auto",
    cmap="bone",
)
ax[1].set(xlabel="z [mm]", ylabel="x [µm]", title="intensity at y = 0", ylim=(-600, 600))
fig.tight_layout()
plt.show()
```

![Intensity of the vortex at the focus, and at y = 0 through the focus](https://raw.githubusercontent.com/melvinrdn/fourierslm/main/docs/_static/readme.png)

## License

MIT
