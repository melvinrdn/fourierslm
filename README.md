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

A Gaussian beam of 1 W on a 1920 × 1200 SLM with 8 µm pixels, a vortex of charge 1, and the field at the focus of a
50 cm lens placed 30 cm after the SLM.

```python
import numpy as np
import fourierslm as fs
from fourierslm import beam, phase, propagate

X, Y = fs.grid(1920, 8e-6), fs.grid(1200, 8e-6)
E = beam.gaussian(X, Y, w=2e-3, power=1.0) * np.exp(1j * phase.vortex(X, Y, charge=1))
x = y = np.linspace(-300e-6, 300e-6, 256)  # window at the focus [m]
E_focus = propagate.focus(E, X, Y, x, y, wavelength=1030e-9, f=0.5, slm_distance=0.3)
I = beam.intensity(E_focus)  # [W/m²]
```

## Speed

On the same 1920 × 1200 SLM, in complex64, on 24 cores. Every propagator gives the analytic Gaussian beam to 10⁻⁸ in
complex128.

| case | run time |
|---|---|
| `focus`, window 256² | 10 ms |
| `focus` on 256², then `free`, 101 planes | 0.2 s |
| `focus_fft`, whole plane (3840 × 2400) | 50 ms |

Details in the [benchmark](https://melvinrdn.github.io/fourierslm/benchmark.html).

## License

MIT
