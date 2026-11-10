# fourierslm

fourierslm computes the field a phase-only spatial light modulator (SLM) makes at the focus of a lens. It builds the
intensity of the beam on the SLM and its phase pattern, then propagates the field to any plane of the focal region, in
physical units.

```python
import fourierslm as fs

X, Y = fs.slm_grid(size=20e-3)  # a square of 20 mm, 2048 × 2048 points
x, y = fs.focus_grid(X, Y, wavelength=1030e-9, f=0.5)

intensity = fs.intensity.gaussian(X, Y, w=2e-3, power=1.0)  # W/m²
phase = fs.phase.vortex(X, Y, charge=1)
E_slm = fs.slm_field(intensity, phase)
E_focus = fs.propagate.focus(E_slm, X, Y, x, y, wavelength=1030e-9, f=0.5)  # |E_focus|² in W/m²
```

Start with the [quickstart](quickstart.md). Every propagator is checked against the analytic Gaussian beam in the
[benchmark](benchmark.md).

```{toctree}
:maxdepth: 2
:hidden:

quickstart
user_guide/index
benchmark
api
development
```
