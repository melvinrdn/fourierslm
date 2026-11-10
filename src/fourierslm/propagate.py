"""Propagation from the SLM, through a lens, to the focal region, and in free space.

The SLM sits at a distance L in front of a lens of focal length f, and z is measured from the back focal plane. With
the ray matrix of this setup, A = −z/f, B = f + z(1 − L/f), D = 1 − L/f, the paraxial field is the Collins integral

    E(x, y, z) = 1/(iλB) · exp[ikD(x² + y²)/(2B)] ∬ E(X, Y) exp[ikA(X² + Y²)/(2B)] exp[−ik(xX + yY)/B] dX dY.

At z = 0 it is the FT of the SLM field times the curvature exp[ik(1 − L/f)(x² + y²)/(2f)], Goodman (5-19). The plane
wave is exp[i(kz − ωt)], fields are complex analytic fields [V/m], and the constant phase exp[ik(L + f + z)] of the
carrier is left out. Arrays have the shape (len(Y), len(X)). Complex64 fields are propagated in complex64, which is
about twice as fast, and FFTs and matrix products run on every core.
"""

import numpy as np
import scipy.fft


def _abd(z, f, slm_distance):
    """Elements A, B, D of the ray matrix from the SLM to the plane z."""
    L = f if slm_distance is None else slm_distance
    return -z / f, f + z * (1 - L / f), 1 - L / f


def _dtype(E):
    """Complex dtype of the computation, complex64 only for single-precision input."""
    return np.result_type(E, np.complex64)


def _expi(phi, dtype):
    """exp(iφ) in dtype, from a float64 phase, by cos and sin (several times faster than a complex exp)."""
    phi = phi.astype(np.finfo(dtype).dtype, copy=False)
    out = np.empty(phi.shape, dtype)
    np.cos(phi, out=out.real)
    np.sin(phi, out=out.imag)
    return out


def _kernel(X, x, k, A, B, D, dtype):
    """Collins kernel along one axis, exp[ik(AX² − 2xX + Dx²)/(2B)], shape (len(X), len(x))."""
    X, x = np.asarray(X, float), np.asarray(x, float)
    phi = np.outer(X, -2 * x)
    phi += A * X[:, None] ** 2
    phi += D * x**2
    phi *= k / (2 * B)
    return _expi(phi, dtype)


def _fft_size(n):
    """Smallest even FFT-friendly size ≥ n."""
    m = scipy.fft.next_fast_len(n)
    while m % 2:
        m = scipy.fft.next_fast_len(m + 1)
    return m


def focus(
    E: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    wavelength: float,
    f: float,
    z: float | np.ndarray = 0.0,
    slm_distance: float | None = None,
) -> np.ndarray:
    """Field in the focal region, on any window and in any plane.

    The Collins integral as two matrix products (matrix Fourier transform), so the window (x, y) is free and no
    padding is needed. Each plane z is exact and independent of the others.

    Parameters
    ----------
    E : ndarray
        Field on the SLM [V/m], shape (len(Y), len(X)).
    X, Y : ndarray
        Uniform 1D coordinates on the SLM [m].
    x, y : ndarray
        1D coordinates of the window [m], any sampling.
    wavelength : float
        Wavelength [m].
    f : float
        Focal length of the lens [m].
    z : float or ndarray, default 0
        Distance from the back focal plane [m], one value or a 1D scan.
    slm_distance : float, optional
        Distance L from the SLM to the lens [m], f by default.

    Returns
    -------
    ndarray
        Field [V/m], shape (len(y), len(x)), or (len(z), len(y), len(x)) for a scan.
    """
    k = 2 * np.pi / wavelength
    dtype = _dtype(E)
    area = (X[1] - X[0]) * (Y[1] - Y[0])
    zs = np.atleast_1d(z)
    out = np.empty((len(zs), len(y), len(x)), dtype)
    for i, zi in enumerate(zs):
        A, B, D = _abd(zi, f, slm_distance)
        Ky = _kernel(Y, y, k, A, B, D, dtype).T
        Kx = _kernel(X, x, k, A, B, D, dtype)
        out[i] = np.linalg.multi_dot([Ky, E, Kx])
        out[i] *= complex(area / (1j * wavelength * B))
    return out[0] if np.ndim(z) == 0 else out


def focus_fft(
    E: np.ndarray,
    pitch: float,
    wavelength: float,
    f: float,
    z: float = 0.0,
    slm_distance: float | None = None,
    pad: float = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Field in a plane of the focal region, on the whole FFT grid.

    The same Collins integral as `focus`, by one FFT of the padded SLM field. Faster than `focus` when the whole plane
    is needed, and the window is fixed by the sampling.

    Parameters
    ----------
    E : ndarray
        Field on the SLM [V/m], on the pixels of `fourierslm.grid`.
    pitch : float
        Pixel pitch of the SLM [m].
    wavelength : float
        Wavelength [m].
    f : float
        Focal length of the lens [m].
    z : float, default 0
        Distance from the back focal plane [m].
    slm_distance : float, optional
        Distance L from the SLM to the lens [m], f by default.
    pad : float, default 2
        Zero padding, the FFT size over the SLM size along each axis, rounded up to an even FFT-friendly size N.

    Returns
    -------
    E_focus : ndarray
        Field [V/m], shape (Ny, Nx).
    x, y : ndarray
        Its coordinates [m], with the pitch λB/(N·pitch) and the axis on index N // 2.
    """
    k = 2 * np.pi / wavelength
    A, B, D = _abd(z, f, slm_distance)
    dtype = _dtype(E)
    (ny, nx), (Ny, Nx) = E.shape, tuple(_fft_size(int(np.ceil(pad * n))) for n in E.shape)

    def factors(n, N):
        """Centering signs and chirps along one axis, before and after the FFT, and the output coordinates."""
        start = N // 2 - n // 2
        X = (np.arange(n) - n // 2) * pitch
        x = (np.arange(N) - N // 2) * wavelength * B / (N * pitch)
        before = (-1.0) ** np.arange(start, start + n) * np.exp(1j * k * A * X**2 / (2 * B))
        after = (-1.0) ** (np.arange(N) + N // 2) * np.exp(1j * k * D * x**2 / (2 * B))
        return start, before.astype(dtype), after.astype(dtype), x

    y0, by, ay, y = factors(ny, Ny)
    x0, bx, ax, x = factors(nx, Nx)

    G = np.zeros((Ny, Nx), dtype)
    view = G[y0 : y0 + ny, x0 : x0 + nx]
    np.multiply(E, by[:, None], out=view)
    view *= bx * dtype.type(pitch**2 / (1j * wavelength * B))
    G = scipy.fft.fft2(G, overwrite_x=True, workers=-1)
    G *= ay[:, None]
    G *= ax
    return G, x, y


def free(E: np.ndarray, pitch: float, wavelength: float, z: float | np.ndarray, bandlimit: bool = True) -> np.ndarray:
    """Free-space propagation by z, angular spectrum on a fixed grid.

    The transfer function is exp[iz(k_z − k)], exact beyond the paraxial approximation, with the evanescent waves
    dropped. The grid is periodic, so keep the field away from its edges. A scan reuses the FFT of E, so it costs one
    inverse FFT per plane.

    Parameters
    ----------
    E : ndarray
        Field [V/m].
    pitch : float
        Pixel pitch [m].
    wavelength : float
        Wavelength [m].
    z : float or ndarray
        Distance [m], negative to propagate backward, one value or a 1D scan.
    bandlimit : bool, default True
        Cut the spatial frequencies the grid cannot sample at this distance, above 1/(λ√(1 + (2z/(N·pitch))²)) along
        each axis, Matsushima and Shimobaba, Opt. Express 17, 19662 (2009).

    Returns
    -------
    ndarray
        Field [V/m] on the same grid, shape (ny, nx) like E, or (len(z), ny, nx) for a scan.
    """
    k = 2 * np.pi / wavelength
    dtype = _dtype(E)
    ny, nx = E.shape
    fx, fy = scipy.fft.fftfreq(nx, pitch)[None, :], scipy.fft.fftfreq(ny, pitch)[:, None]
    s2 = (wavelength * fx) ** 2 + (wavelength * fy) ** 2
    propagating = s2 < 1
    kz_minus_k = -k * s2 / (1 + np.sqrt(np.where(propagating, 1 - s2, 1)))  # without the cancellation of k_z − k
    G = scipy.fft.fft2(E, workers=-1)
    zs = np.atleast_1d(z)
    out = np.empty((len(zs), ny, nx), dtype)
    for i, zi in enumerate(zs):
        keep = propagating
        if bandlimit:
            keep = keep & (wavelength * np.abs(fx) * np.hypot(1, 2 * zi / (nx * pitch)) <= 1)
            keep &= wavelength * np.abs(fy) * np.hypot(1, 2 * zi / (ny * pitch)) <= 1
        H = _expi(zi * kz_minus_k, dtype)
        H *= keep
        H *= G
        out[i] = scipy.fft.ifft2(H, overwrite_x=True, workers=-1)
    return out[0] if np.ndim(z) == 0 else out
