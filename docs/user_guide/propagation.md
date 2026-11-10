# Propagation

## Setup

The SLM sits at a distance $L$ in front of a lens of focal length $f$. In the back focal plane, the field is the
Fourier transform (FT) of the field on the SLM, times a curvature [1],

$$
E(x, y) = \frac{1}{i\lambda f}\, e^{ik(1 - L/f)(x^2 + y^2)/(2f)} \iint E(X, Y)\, e^{-ik(xX + yY)/f}\,
\mathrm{d}X\, \mathrm{d}Y = \frac{k}{if}\, e^{ik(1 - L/f)(x^2 + y^2)/(2f)}\,
E\!\left(k_X = \frac{kx}{f},\, k_Y = \frac{ky}{f}\right),
$$

with the unitary FT $E(k_X, k_Y) = \frac{1}{2\pi} \iint E(X, Y)\, e^{-i(k_X X + k_Y Y)}\, \mathrm{d}X\, \mathrm{d}Y$.
The constant phase $e^{ik(f + L)}$ is left out. The curvature vanishes only when the SLM sits in the front focal plane,
$L = f$, the default. Over a single focus it is small, but across a pattern of several foci or a ring it is not.

## Functions

| function | gives | method |
|---|---|---|
| `focus(E, pitch, wavelength, f, slm_distance, pad)` | the field in the back focal plane, with its coordinates | FFT of the padded SLM |
| `free(E, pitch, wavelength, z)` | the field after free propagation, in one plane or a scan in z | angular spectrum |

`focus` evaluates the FT by one FFT of the SLM field, zero-padded to $N \times N$, with $N$ equal to `pad` times the
larger side of the SLM, 2 by default. The grid spans $\lambda f/p$, the whole plane the SLM reaches with pixels of pitch
$p$, and its pitch is $\lambda f/(Np)$ along $x$ and $y$.

`free` uses the exact transfer function $e^{iz(k_z - k)}$, beyond the paraxial approximation, on a fixed periodic grid.
Keep the field away from the edges of the grid over the whole propagation.

## Other planes

To follow the beam through the focus, take `focus`, keep the part of the plane around the beam, and give it to `free`
with an array of z, measured from the back focal plane. It costs one inverse FFT per plane. The part kept must hold the
beam over the whole scan.

## Example

Left, a vortex of charge 1 through the focus, by `focus` and `free`. Right, the phase of a Gaussian focus for two
distances from the SLM to the lens. It is flat for $L = f$ and curved otherwise.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   from fourierslm import beam, grid, phase, propagate

   lam, f, w, pitch = 1030e-9, 0.5, 2e-3, 8e-6
   X, Y = grid(1920, pitch), grid(1200, pitch)
   E = beam.gaussian(X, Y, w).astype(np.complex64)
   z_R = lam * f**2 / (np.pi * w**2)

   vortex, u, _ = propagate.focus(E * np.exp(1j * phase.vortex(X, Y, 1)), pitch, lam, f)
   keep = np.abs(u) < 1e-3  # x and y share the grid
   u = u[keep]
   z = np.linspace(-3, 3, 121) * z_R
   scan = propagate.free(vortex[np.ix_(keep, keep)], u[1] - u[0], lam, z)

   fig, ax = plt.subplots(1, 2)
   ax[0].imshow(beam.intensity(scan[:, len(u) // 2, :]).T, extent=[z[0] * 1e3, z[-1] * 1e3, u[0] * 1e6, u[-1] * 1e6],
                aspect="auto", origin="lower")
   ax[0].set(xlabel="z [mm]", ylabel="x [µm]", title="vortex, intensity at y = 0", ylim=(-500, 500))
   for L in (f, 0.3):
       E_focus, x, y = propagate.focus(E, pitch, lam, f, slm_distance=L)
       row, near = E_focus[len(y) // 2], np.abs(x) <= 200e-6
       ax[1].plot(x[near] * 1e6, np.angle(row[near] / row[len(x) // 2]), label=f"L = {L * 100:.0f} cm")
   ax[1].set(xlabel="x [µm]", ylabel="phase at the focus [rad]", title="Gaussian, y = 0")
   ax[1].legend()
   fig.tight_layout()
```

## References

1. J. W. Goodman, *Introduction to Fourier Optics*, 2nd ed., McGraw-Hill (1996), Eq. (5-19).
