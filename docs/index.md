# fourierslm

fourierslm computes the field a phase-only spatial light modulator (SLM) makes at the focus of a lens. It builds the
beam on the SLM and its phase pattern, then propagates the field to any plane of the focal region, in physical units.

```python
import numpy as np
from fourierslm import beam, grid, phase, propagate

X, Y = grid(1920, 8e-6), grid(1200, 8e-6)  # a 1920 × 1200 SLM with 8 µm pixels
E = beam.gaussian(X, Y, w=2e-3, power=1.0) * np.exp(1j * phase.vortex(X, Y, charge=1))  # field [V/m]
E_focus, x, y = propagate.focus(E, pitch=8e-6, wavelength=1030e-9, f=0.5)  # back focal plane
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
