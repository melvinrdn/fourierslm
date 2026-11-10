"""Fourier optics of a phase-only SLM: input beams, phase patterns and propagation to the focus of a lens."""

import numpy as np

from . import beam, phase, propagate

__all__ = ["beam", "grid", "phase", "propagate"]


def grid(n: int, pitch: float) -> np.ndarray:
    """Centered pixel coordinates, with the axis on pixel n // 2.

    The same convention as `propagate.focus_fft`, so the phase patterns line up with its output.

    Parameters
    ----------
    n : int
        Number of pixels.
    pitch : float
        Pixel pitch [m].

    Returns
    -------
    ndarray
        Coordinates (i − n // 2)·pitch of the pixels i = 0 … n − 1 [m].
    """
    return (np.arange(n) - n // 2) * pitch
