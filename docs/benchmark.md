# Benchmark

A Gaussian beam with a flat wavefront on the SLM stays Gaussian through the lens, so the Collins integral has a closed
form. With $q_\mathrm{in} = -i\pi w^2/\lambda$ and the ray matrix of the [setup](user_guide/propagation.md),

$$
E(r, z) = \frac{E_0}{A + B/q_\mathrm{in}}\, e^{ikr^2/(2q)}, \qquad q = \frac{Aq_\mathrm{in} + B}{Cq_\mathrm{in} + D},
\qquad C = -\frac{1}{f}.
$$

Every propagator is compared with this field, with the SLM in the front focal plane and 20 cm closer to the lens.

## Summary

- `focus` and `focus_fft` give the analytic field to 10⁻⁸ in complex128, the part of the beam cut by the edge of the
  SLM, and to a few 10⁻⁶ in complex64.
- `focus` followed by `free` agrees to 10⁻⁶ over ±4 Rayleigh ranges. `free` is exact beyond the paraxial
  approximation, and this is the size of the correction.
- On a 1920 × 1200 SLM, `focus` takes about 10 ms for a 256² window in complex64, and a scan of 100 planes by `focus`
  and `free` about 0.2 s.

## Figure

Left, the intensity through the focus, with the analytic radius $w(z)$. Middle and right, the intensity and the phase
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

## Tables

```{include} benchmark_results.md
```
