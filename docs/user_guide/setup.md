# The SLM

## Grid

The SLM is a grid of square pixels. `grid(n, pitch)` gives the coordinates of its `n` pixels along one axis, with
the optical axis on pixel `n // 2`. The two axes are separate, so the grid can be rectangular like the SLM itself.

```python
X, Y = grid(1920, 8e-6), grid(1200, 8e-6)
```

## Beam

The `beam` module gives the field that falls on the SLM, not its intensity, with a flat wavefront and a given power.

| function | field |
|---|---|
| `gaussian(X, Y, w, power)` | Gaussian of radius `w` at 1/e² of the intensity |
| `flattop(X, Y, radius, power)` | uniform disk |
| `intensity(E)` | $\frac{1}{2}\varepsilon_0 c\vert E\vert^2$ |

## Phase patterns

The `phase` module gives the phase the SLM writes on the beam. The patterns are continuous, so they add, and the field
after the SLM is the beam times $e^{i\phi}$.

| function | phase |
|---|---|
| `lens(X, Y, wavelength, f)` | $-k(X^2 + Y^2)/(2f)$ |
| `axicon(X, Y, wavelength, angle)` | $-kR\sin\alpha$ |
| `grating(X, Y, wavelength, angle, azimuth)` | $k\sin\theta\,(X\cos\psi + Y\sin\psi)$ |
| `vortex(X, Y, charge)` | $\ell\psi$ |
| `binary(X, Y, period, depth)` | stripes of 0 and `depth` |
| `radial_step(X, Y, radius, inner, outer)` | `inner` inside the radius, `outer` outside |

`wrap` brings the sum into [0, 2π) and `quantize` rounds it to the gray levels of the SLM, for display. Neither
changes the field, since $e^{i\phi}$ only sees the phase modulo 2π.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   from fourierslm import grid, phase

   lam = 1030e-9
   X = Y = grid(400, 8e-6)
   patterns = {
       "lens, f = 1 m": phase.lens(X, Y, lam, 1.0),
       "axicon, 1 mrad": phase.axicon(X, Y, lam, 1e-3),
       "grating, 0.5 mrad at 30°": phase.grating(X, Y, lam, 5e-4, np.pi / 6),
       "vortex, charge 3": phase.vortex(X, Y, 3),
       "binary, period 0.4 mm": phase.binary(X, Y, 0.4e-3),
       "vortex + lens, 4 bits": phase.quantize(phase.vortex(X, Y, 1) + phase.lens(X, Y, lam, 1.0), bits=4),
   }
   fig, ax = plt.subplots(1, 6, figsize=(12, 2.4))
   for a, (title, p) in zip(ax, patterns.items()):
       a.imshow(phase.wrap(p), extent=[X[0] * 1e3, X[-1] * 1e3, Y[0] * 1e3, Y[-1] * 1e3], origin="lower",
                cmap="twilight", vmin=0, vmax=2 * np.pi)
       a.set(title=title, xticks=[], yticks=[])
       a.title.set_fontsize(8)
   fig.tight_layout()
```
