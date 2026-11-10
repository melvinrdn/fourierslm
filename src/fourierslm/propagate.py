"""Propagation from the SLM to the back focal plane of a lens, and in free space.

The SLM sits at a distance L in front of a lens of focal length f. In the back focal plane the field is the FT of the
field on the SLM times a curvature, Goodman (5-19),

    E(x, y) = 1/(iλf) · exp[ik(1 − L/f)(x² + y²)/(2f)] ∬ E(X, Y) exp[−ik(xX + yY)/f] dX dY
            = k/(if) · exp[ik(1 − L/f)(x² + y²)/(2f)] · E(k_X = kx/f, k_Y = ky/f),

with the unitary FT E(k_X, k_Y) = 1/(2π) ∬ E(X, Y) exp[−i(k_X X + k_Y Y)] dX dY. The other planes follow by `free`.
The plane wave is exp[i(kz − ωt)], fields are in √(W/m²) so that ``abs(E)**2`` is the intensity, and the constant phase
exp[ik(L + f)] is left out. Grids are square, and arrays have the shape (n, n). Complex64 fields are propagated in
complex64, which is about twice as fast, and FFTs run on every core.
"""

import numpy as np
import scipy.fft


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
    slm_distance: float | None = None,
) -> np.ndarray:
    """Field in the back focal plane of the lens, on the grid of `fourierslm.focus_grid`.

    The FT of the field on the SLM, with the prefactor 1/(iλf) and the curvature exp[ik(1 − L/f)(x² + y²)/(2f)], by one
    FFT of the SLM field zero-padded to N × N. N is read from the pitch λf/(N·pitch) of x, so the padding is chosen
    once, in `fourierslm.focus_grid`.

    Parameters
    ----------
    E : ndarray
        Field on the SLM [√(W/m²)], shape (n, n).
    X, Y : ndarray
        Coordinates on the SLM [m], from `fourierslm.slm_grid`.
    x, y : ndarray
        Coordinates at the focus [m], from `fourierslm.focus_grid`, with or without a window.
    wavelength : float
        Wavelength [m].
    f : float
        Focal length of the lens [m].
    slm_distance : float, optional
        Distance L from the SLM to the lens [m], f by default.

    Returns
    -------
    ndarray
        Field [√(W/m²)], shape (len(y), len(x)).

    Raises
    ------
    ValueError
        If the grids are not square, or x is not a focal grid of X at this wavelength and focal length.
    """
    if not (np.array_equal(X, Y) and np.array_equal(x, y)):
        raise ValueError("the grids must be square, Y = X and y = x")
    n, m = len(X), len(x)
    pitch = (X[-1] - X[0]) / (n - 1)
    N = round(wavelength * f / (pitch * (x[1] - x[0])))
    grid = (np.arange(m) - m // 2) * wavelength * f / (N * pitch)
    if N % 2 or N < max(n, m) or not np.allclose(x, grid, rtol=1e-9, atol=0):
        raise ValueError("x is not a focal grid of X at this wavelength and focal length, make it with focus_grid")
    k = 2 * np.pi / wavelength
    L = f if slm_distance is None else slm_distance
    dtype = _dtype(E)
    a, b = N // 2 - n // 2, N // 2 - m // 2  # first SLM pixel and first focal sample in the N × N grid
    sign = (-1.0) ** np.arange(a, a + n)  # with the signs after the FFT, puts the axis on index N // 2 in both planes
    after = ((-1.0) ** (np.arange(b, b + m) + N // 2) * np.exp(1j * k * (1 - L / f) * x**2 / (2 * f))).astype(dtype)

    G = np.zeros((N, N), dtype)
    view = G[a : a + n, a : a + n]
    np.multiply(E, sign[:, None], out=view)
    view *= (sign * pitch**2 / (1j * wavelength * f)).astype(dtype)
    G = scipy.fft.fft2(G, overwrite_x=True, workers=-1)[b : b + m, b : b + m]
    return G * after[:, None] * after


def free(E: np.ndarray, pitch: float, wavelength: float, z: float | np.ndarray, bandlimit: bool = True) -> np.ndarray:
    """Free-space propagation by z, angular spectrum on a fixed grid.

    The transfer function is exp[iz(k_z − k)], exact beyond the paraxial approximation, with the evanescent waves
    dropped. The grid is periodic, so keep the field away from its edges. A scan reuses the FFT of E, so it costs one
    inverse FFT per plane.

    Parameters
    ----------
    E : ndarray
        Field [√(W/m²)].
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
        Field [√(W/m²)] on the same grid, shape (ny, nx) like E, or (len(z), ny, nx) for a scan.
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
