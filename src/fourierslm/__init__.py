"""Fourier optics of a phase-only SLM: beam intensities, phase patterns and propagation to the focus of a lens."""

import numpy as np

from . import intensity, phase, propagate
from .propagate import _fft_size

__all__ = ["focus_grid", "intensity", "peak_power", "phase", "propagate", "slm_field", "slm_grid"]


def slm_grid(size: float | None = None, n: int = 2048, *, pitch: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates on the SLM, a square of n × n pixels with the axis on pixel n // 2.

    Choosing the grid. The size must hold the beam, at least 4 beam radii for a Gaussian. With the size fixed, the
    number of points sets the field of view at the focus, n·λf/(2·size). A bigger field of view needs more points. With
    n fixed, the size sets the detail at the focus, λf/(2·size). Finer detail needs a bigger size, and 20 times the beam
    radius gives about 0.08·w₀. Give `pitch` instead of `size` only when the SLM pixels matter, as for stochastic
    holograms or quantization. Then size = n·pitch. The values at the focus are for the default pad 2 of `focus_grid`.

    Parameters
    ----------
    size : float, optional
        Side of the square [m]. Give it or `pitch`.
    n : int, default 2048
        Number of pixels along each side.
    pitch : float, optional
        Pixel pitch [m], instead of `size`.

    Returns
    -------
    X, Y : ndarray
        Coordinates (i − n // 2)·pitch of the pixels i = 0 … n − 1 [m], X = Y.

    Raises
    ------
    ValueError
        If both or neither of `size` and `pitch` are given.
    """
    if (size is None) == (pitch is None):
        raise ValueError("give size or pitch, not both")
    X = (np.arange(n) - n // 2) * (size / n if pitch is None else pitch)
    return X, X.copy()


def focus_grid(
    X: np.ndarray, Y: np.ndarray, wavelength: float, f: float, pad: float = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates at the focus of the lens, on which `propagate.focus` gives the field.

    The pitch is λf/(N·pitch), with N the side of the zero-padded FFT, pad·n rounded up to an even FFT-friendly size.
    The n samples around the axis are kept, so the window is n·λf/(N·pitch). With pad 2, the pitch is λf/(2·size) and
    the window is the central half of the plane the SLM reaches, λf/pitch. With pad 1, it is the whole plane at twice
    the pitch.

    Parameters
    ----------
    X, Y : ndarray
        Coordinates on the SLM [m], from `slm_grid`.
    wavelength : float
        Wavelength [m].
    f : float
        Focal length of the lens [m].
    pad : float, default 2
        Zero padding of the FFT, N over n.

    Returns
    -------
    x, y : ndarray
        Coordinates [m], x = y, n samples with the axis on index n // 2.

    Raises
    ------
    ValueError
        If the grid is not square.
    """
    if not np.array_equal(X, Y):
        raise ValueError("the grid must be square, Y = X")
    n = len(X)
    pitch = (X[-1] - X[0]) / (n - 1)
    N = _fft_size(int(np.ceil(pad * n)))
    x = (np.arange(n) - n // 2) * (wavelength * f / (N * pitch))
    return x, x.copy()


def slm_field(intensity: np.ndarray, phase: np.ndarray | float) -> np.ndarray:
    """Field on the SLM, √I·exp(iφ).

    Fields are in √(W/m²), so ``abs(E)**2`` is the intensity [W/m²], on the SLM and in every plane after it.

    Parameters
    ----------
    intensity : ndarray
        Intensity of the beam [W/m²], from the `intensity` module.
    phase : ndarray or float
        Phase written by the SLM [rad], from the `phase` module.

    Returns
    -------
    ndarray
        Complex field [√(W/m²)].
    """
    return np.sqrt(intensity) * np.exp(1j * np.asarray(phase))


def peak_power(energy: float, duration: float) -> float:
    """Peak power of a Gaussian pulse, √(4 ln 2/π)·energy/duration ≈ 0.94·energy/duration.

    Give it as the power of the `intensity` functions to get the peak intensity.

    Parameters
    ----------
    energy : float
        Energy of the pulse [J].
    duration : float
        FWHM of the intensity in time [s].

    Returns
    -------
    float
        Peak power [W].
    """
    return np.sqrt(4 * np.log(2) / np.pi) * energy / duration
