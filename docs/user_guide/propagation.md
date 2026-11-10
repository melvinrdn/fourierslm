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
| `focus_grid(X, Y, wavelength, f, pad, window)` | the coordinates in the back focal plane | |
| `focus(E, X, Y, x, y, wavelength, f, slm_distance)` | the field on (x, y) | FFT of the padded SLM |
| `free(E, pitch, wavelength, z)` | the field after free propagation, in one plane or a scan in z | angular spectrum |

`focus` evaluates the FT by one FFT of the SLM field, zero-padded to $N \times N$. `focus_grid` sets $N$, `pad` times
the number of points n of the SLM, 2 by default, and keeps the n samples around the axis. Their pitch is
$\lambda f/(Np)$ for pixels of pitch $p$. With `pad=2`, the window is the central half of the plane the SLM reaches,
$\lambda f/p$. With `pad=1`, it is the whole plane at twice the pitch. `focus` reads $N$ back from the pitch of x, so
the padding is chosen once. It raises an error if x is not a focal grid of X at this wavelength and focal length.

`focus_grid(..., window=a)` keeps only a square of side `a` around the axis, and `focus` then gives the field on that
window only.

`free` uses the exact transfer function $e^{iz(k_z - k)}$, beyond the paraxial approximation, on a fixed periodic grid.
Keep the field away from the edges of the grid over the whole propagation.

## Other planes

To follow the beam through the focus, compute `focus` on a window of the focal grid around the beam, and give it to
`free` with an array of z, measured from the back focal plane. It costs one inverse FFT per plane. The part kept
must hold the beam over the whole scan.

## Example

Left, a vortex of charge 1 through the focus, by `focus` and `free`. Right, the phase of a Gaussian focus for two
distances from the SLM to the lens. It is flat for $L = f$ and curved otherwise.

```{eval-rst}
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np

   import fourierslm as fs

   lam, f, w = 1030e-9, 0.5, 2e-3
   X, Y = fs.slm_grid(size=10 * w)
   x, y = fs.focus_grid(X, Y, lam, f)
   intensity = fs.intensity.gaussian(X, Y, w)
   z_R = lam * f**2 / (np.pi * w**2)

   u, _ = fs.focus_grid(X, Y, lam, f, window=2e-3)  # ±1 mm around the axis
   E_slm = fs.slm_field(intensity, fs.phase.vortex(X, Y, 1)).astype(np.complex64)
   z = np.linspace(-3, 3, 121) * z_R
   scan = fs.propagate.free(fs.propagate.focus(E_slm, X, Y, u, u, lam, f), u[1] - u[0], lam, z)

   fig, ax = plt.subplots(1, 2)
   ax[0].imshow(np.abs(scan[:, len(u) // 2, :]).T ** 2, extent=[z[0] * 1e3, z[-1] * 1e3, u[0] * 1e6, u[-1] * 1e6],
                aspect="auto", origin="lower")
   ax[0].set(xlabel="z [mm]", ylabel="x [µm]", title="vortex, intensity at y = 0", ylim=(-500, 500))
   for L in (f, 0.3):
       E_focus = fs.propagate.focus(fs.slm_field(intensity, 0), X, Y, x, y, lam, f, slm_distance=L)
       row, near = E_focus[len(y) // 2], np.abs(x) <= 200e-6
       ax[1].plot(x[near] * 1e6, np.angle(row[near] / row[len(x) // 2]), label=f"L = {L * 100:.0f} cm")
   ax[1].set(xlabel="x [µm]", ylabel="phase at the focus [rad]", title="Gaussian, y = 0")
   ax[1].legend()
   fig.tight_layout()
```

## References

1. J. W. Goodman, *Introduction to Fourier Optics*, 2nd ed., McGraw-Hill (1996), Eq. (5-19).
