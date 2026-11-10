# Propagation

## Setup

The SLM sits at a distance $L$ in front of a lens of focal length $f$, and $z$ is measured from the back focal plane.
From the SLM to the plane $z$, the ray matrix has the elements

$$
A = -\frac{z}{f}, \qquad B = f + z\left(1 - \frac{L}{f}\right), \qquad D = 1 - \frac{L}{f},
$$

and the paraxial field is the Collins integral [1]

$$
E(x, y, z) = \frac{1}{i\lambda B}\, e^{ikD(x^2 + y^2)/(2B)} \iint E(X, Y)\, e^{ikA(X^2 + Y^2)/(2B)}\,
e^{-ik(xX + yY)/B}\, \mathrm{d}X\, \mathrm{d}Y.
$$

At $z = 0$ it is the Fourier transform of the field on the SLM, times the curvature
$e^{ik(1 - L/f)(x^2 + y^2)/(2f)}$ [2]. The curvature vanishes only when the SLM sits in the front focal plane,
$L = f$, the default. Over a single focus it is small, but across a pattern of several foci or a ring it is not.

## Functions

| function | gives | method |
|---|---|---|
| `focus(E, X, Y, x, y, wavelength, f, z, slm_distance)` | the field on any window (x, y), in one plane or a scan in z | matrix Fourier transform |
| `focus_fft(E, pitch, wavelength, f, z, slm_distance, pad)` | the field on the whole FFT grid, with its coordinates | FFT of the padded SLM |
| `free(E, pitch, wavelength, z)` | the field after free propagation, in one plane or a scan in z | angular spectrum |

`focus` evaluates the Collins integral as two matrix products, so the window and its sampling are free, for instance
the pixels of a camera, and no padding is needed. Its cost grows with the size of the window. `focus_fft` gives the
whole plane at once, with the pitch $\lambda B/(N p)$ for an FFT of size $N$ and pixels of pitch $p$, so `pad` sets the
sampling. Both give the same field to 10⁻¹⁰.

`free` uses the exact transfer function $e^{iz(k_z - k)}$, beyond the paraxial approximation, on a fixed periodic grid.
Keep the field away from the edges of the grid over the whole propagation.

## Scans in z

There are two ways to follow the beam through the focus.

- `focus` with an array of z computes each plane exactly from the SLM, and each plane could have its own window. It
  costs one `focus` per plane, 10 to 20 ms on a 256² window from a 1920 × 1200 SLM.
- `focus` once at $z = 0$, then `free` with an array of z, costs one inverse FFT per plane, about ten times less. The
  window must hold the beam over the whole scan.

## Example

Left, a vortex of charge 1 through the focus, by `focus` on a window of 512² and `free`. Right, the phase of a
Gaussian focus for two distances from the SLM to the lens. It is flat for $L = f$ and curved otherwise.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   import fourierslm as fs
   from fourierslm import beam, phase, propagate

   lam, f, w = 1030e-9, 0.5, 2e-3
   X, Y = fs.grid(1920, 8e-6), fs.grid(1200, 8e-6)
   E = beam.gaussian(X, Y, w).astype(np.complex64)
   z_R = lam * f**2 / (np.pi * w**2)

   u = fs.grid(512, 2e-6)
   z = np.linspace(-3, 3, 121) * z_R
   vortex = E * np.exp(1j * phase.vortex(X, Y, 1))
   scan = propagate.free(propagate.focus(vortex, X, Y, u, u, lam, f), 2e-6, lam, z)

   x = np.linspace(-200e-6, 200e-6, 401)
   fig, ax = plt.subplots(1, 2)
   ax[0].imshow(beam.intensity(scan[:, 256, :]).T, extent=[z[0] * 1e3, z[-1] * 1e3, u[0] * 1e6, u[-1] * 1e6],
                aspect="auto", origin="lower")
   ax[0].set(xlabel="z [mm]", ylabel="x [µm]", title="vortex, intensity at y = 0")
   for L in (f, 0.3):
       E_focus = propagate.focus(E, X, Y, x, np.zeros(1), lam, f, slm_distance=L)[0]
       ax[1].plot(x * 1e6, np.angle(E_focus / E_focus[200]), label=f"L = {L * 100:.0f} cm")
   ax[1].set(xlabel="x [µm]", ylabel="phase at the focus [rad]", title="Gaussian, y = 0")
   ax[1].legend()
   fig.tight_layout()
```

## References

1. S. A. Collins, J. Opt. Soc. Am. **60**, 1168 (1970).
2. J. W. Goodman, *Introduction to Fourier Optics*, 2nd ed., McGraw-Hill (1996), Eq. (5-19).
