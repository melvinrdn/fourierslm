"""Input beams on the SLM.

Fields are complex analytic fields [V/m], so the intensity is I = ½ε₀c|E|². Coordinates X and Y are 1D, and the
arrays have the shape (len(Y), len(X)).
"""

import numpy as np
from scipy.constants import c, epsilon_0


def intensity(E: np.ndarray) -> np.ndarray:
    """Intensity of a field.

    Parameters
    ----------
    E : ndarray
        Complex field [V/m].

    Returns
    -------
    ndarray
        I = ½ε₀c|E|² [W/m²].
    """
    return 0.5 * epsilon_0 * c * np.abs(E) ** 2


def _amplitude(I0):
    """Field amplitude of the intensity I0."""
    return np.sqrt(2 * I0 / (epsilon_0 * c))


def _r2(X, Y, center):
    """Squared distance to center, shape (len(Y), len(X))."""
    return (np.asarray(X)[None, :] - center[0]) ** 2 + (np.asarray(Y)[:, None] - center[1]) ** 2


def gaussian(
    X: np.ndarray, Y: np.ndarray, w: float, power: float = 1.0, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Field of a Gaussian beam with a flat wavefront, not its intensity.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    w : float
        Radius at 1/e² of the intensity [m].
    power : float, default 1
        Power [W].
    center : tuple of float, default (0, 0)
        Center of the beam [m].

    Returns
    -------
    ndarray
        Real field E [V/m], with I = 2P/(πw²)·exp(−2r²/w²).
    """
    return _amplitude(2 * power / (np.pi * w**2)) * np.exp(-_r2(X, Y, center) / w**2)


def flattop(
    X: np.ndarray, Y: np.ndarray, radius: float, power: float = 1.0, center: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Field of a uniform beam cut by a circular aperture, not its intensity.

    Parameters
    ----------
    X, Y : ndarray
        1D coordinates [m].
    radius : float
        Radius of the aperture [m].
    power : float, default 1
        Power [W] of the continuous disk. The pixelated edge changes it slightly.
    center : tuple of float, default (0, 0)
        Center of the beam [m].

    Returns
    -------
    ndarray
        Real field E [V/m], with I = P/(π radius²) inside the disk and 0 outside.
    """
    return _amplitude(power / (np.pi * radius**2)) * (_r2(X, Y, center) <= radius**2)
