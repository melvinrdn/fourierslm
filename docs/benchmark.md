# Benchmark

A Gaussian beam of radius $w$ with a flat wavefront on the SLM gives in the back focal plane, by the
[FT formula](user_guide/propagation.md), a Gaussian of radius $w_0 = \lambda f/(\pi w)$ with the curvature
$(1 - L/f)/f$. Free space adds $z$ to its complex radius

$$
\frac{1}{q_0} = \frac{1}{f}\left(1 - \frac{L}{f}\right) + \frac{2i}{kw_0^2},
$$

so that at the distance $z$ from the back focal plane

$$
E(r, z) = E_0\, \frac{q_0}{q_0 + z}\, e^{ikr^2/[2(q_0 + z)]},
$$

with $E_0 = -i\sqrt{I_0}$ the on-axis field in the back focal plane and $I_0 = 2P/(\pi w_0^2)$.

`focus` is compared with this field in the back focal plane, and `focus` followed by `free` over the scan, with the SLM
in the front focal plane and 20 cm closer to the lens.

## Summary

- `focus` gives the analytic field in the back focal plane to 3·10⁻⁸ in complex128, the part of the beam cut by the
  edge of the SLM, and to 2·10⁻⁷ in complex64.
- `focus` followed by `free` agrees to 10⁻⁶ over ±4 Rayleigh ranges. `free` is exact beyond the paraxial
  approximation, and this is the size of the correction.

## Figure

Left, the intensity through the focus by `focus` and `free`, with the analytic radius $w(z)$. Middle and right, the intensity and the phase
on the axis, for both distances from the SLM to the lens. The on-axis phase is the Gouy phase, shifted by the
curvature when $L \neq f$.

```{image} _static/benchmark/gaussian.png
:alt: Intensity map through the focus, on-axis intensity and on-axis phase, fourierslm against the analytic beam
```

## Running the benchmark

From the root of the repository,

```bash
uv run --with matplotlib python benchmarks/analytic/run.py
```

## Table

```{include} benchmark_results.md
```
