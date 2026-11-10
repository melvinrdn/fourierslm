import numpy as np

import fourierslm as fs
from fourierslm import beam

PITCH = 8e-6
X, Y = fs.grid(512, PITCH), fs.grid(384, PITCH)


def test_grid():
    x = fs.grid(6, 2.0)
    assert np.array_equal(x, [-6, -4, -2, 0, 2, 4])
    assert fs.grid(5, 1.0)[2] == 0


def test_gaussian():
    E = beam.gaussian(X, Y, 0.3e-3, power=2.0, center=(40e-6, -80e-6))
    I = beam.intensity(E)
    assert E.shape == (384, 512)
    assert np.isclose(I.sum() * PITCH**2, 2.0, rtol=1e-9)
    assert np.isclose(I[192 - 10, 256 + 5], 2 * 2.0 / (np.pi * 0.3e-3**2))


def test_flattop():
    E = beam.flattop(X, Y, 1e-3, power=0.5)
    I = beam.intensity(E)
    assert np.isclose(I.max(), 0.5 / (np.pi * 1e-3**2))
    assert np.isclose(I.sum() * PITCH**2, 0.5, rtol=1e-2)
    assert I[192, 256 + 126] == 0 and I[192, 256 + 125] > 0
