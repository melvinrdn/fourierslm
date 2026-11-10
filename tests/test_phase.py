import numpy as np
import pytest

from fourierslm import focus_grid, intensity, phase, propagate, slm_field, slm_grid

LAM, F, PITCH = 1030e-9, 0.2, 8e-6
X, Y = slm_grid(n=512, pitch=PITCH)


def test_lens_sign():
    """A converging lens delays the edge, −k r²/(2f)."""
    p = phase.lens(X, Y, LAM, F, center=(1e-4, 0.0))
    assert p.shape == (512, 512)
    assert np.isclose(p[256, 256 + 100], -np.pi / (LAM * F) * (100 * PITCH - 1e-4) ** 2)


@pytest.mark.parametrize(("azimuth", "axis"), [(0.0, "x"), (np.pi / 2, "y")])
def test_grating_steers_the_focus(azimuth, axis):
    angle = 2e-4
    E = slm_field(intensity.gaussian(X, Y, 0.4e-3), phase.grating(X, Y, LAM, angle, azimuth))
    x, y = focus_grid(X, Y, LAM, F)
    I = np.abs(propagate.focus(E, X, Y, x, y, LAM, F)) ** 2
    centroid = np.sum(I * (x[None, :] if axis == "x" else y[:, None])) / I.sum()
    assert np.isclose(centroid, F * np.sin(angle), rtol=1e-3)


def test_vortex_winds_once_per_charge():
    p = phase.vortex(X, Y, 3)
    t = np.linspace(0, 2 * np.pi, 400)
    ring = p[(256 + 50 * np.sin(t)).astype(int), (256 + 50 * np.cos(t)).astype(int)]
    assert np.isclose(np.sum(np.angle(np.exp(1j * np.diff(ring)))), 6 * np.pi)


def test_vortex_rotation_moves_the_step():
    p = phase.vortex(X, Y, 0.5, rotation=np.pi / 2)
    assert np.isclose(p[256 - 50, 256], 0.5 * np.pi, atol=1e-12) or np.isclose(p[256 - 50, 256], -0.5 * np.pi)
    assert np.isclose(p[256 + 50, 256], 0.0, atol=1e-12)


def test_axicon():
    p = phase.axicon(X, Y, LAM, 1e-3)
    assert np.isclose(p[256 + 30, 256 - 40], -2 * np.pi / LAM * np.sin(1e-3) * 50 * PITCH)


def test_binary():
    p = phase.binary(X, Y, period=10 * PITCH, depth=2.0)
    assert set(np.unique(p)) == {0.0, 2.0}
    assert np.array_equal(p[0, 256:266], [0, 0, 0, 0, 0, 2, 2, 2, 2, 2])
    assert np.array_equal(phase.binary(X, Y, 10 * PITCH, azimuth=np.pi / 2), phase.binary(X, Y, 10 * PITCH).T)


def test_radial_step():
    p = phase.radial_step(X, Y, 20 * PITCH, inner=1.0, outer=3.0)
    assert p[256, 256 + 20] == 1.0 and p[256, 256 + 21] == 3.0


def test_wrap():
    p = phase.wrap(np.array([-0.5, 0.0, 2 * np.pi, 7.0]))
    assert np.allclose(p, [2 * np.pi - 0.5, 0.0, 0.0, 7.0 - 2 * np.pi])


def test_quantize():
    p = phase.quantize(np.linspace(-10, 10, 10001), bits=3)
    assert np.allclose(np.unique(p), 2 * np.pi / 8 * np.arange(8))
    assert phase.quantize(np.array([2 * np.pi - 1e-3]), bits=8)[0] == 0.0
