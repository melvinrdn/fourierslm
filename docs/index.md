# fourierslm

fourierslm computes the field a phase-only spatial light modulator (SLM) makes at the focus of a lens. It builds the
beam on the SLM and its phase pattern, then propagates the field to any plane of the focal region, in physical units.

```python
import numpy as np
import fourierslm as fs
from fourierslm import beam, phase, propagate

X, Y = fs.grid(1920, 8e-6), fs.grid(1200, 8e-6)  # a 1920 × 1200 SLM with 8 µm pixels
E = beam.gaussian(X, Y, w=2e-3, power=1.0) * np.exp(1j * phase.vortex(X, Y, charge=1))
x = y = np.linspace(-300e-6, 300e-6, 256)  # window at the focus [m]
E_focus = propagate.focus(E, X, Y, x, y, wavelength=1030e-9, f=0.5)
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
