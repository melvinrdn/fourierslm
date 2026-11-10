### Accuracy

Gaussian beam of radius 1.2 mm on a 1920 × 1200 SLM of 8 µm pixels, 1030 nm, f = 50 cm, so w₀ = 137 µm and z_R = 5.7 cm. Largest difference over a window of ±3w₀, relative to the peak.

| SLM–lens distance | method | precision | planes | error |
|---|---|---|---|---|
| 50 cm | `focus` | complex128 | z from −4 z_R to 4 z_R | 1.5e-08 |
| 50 cm | `focus` | complex64 | z from −4 z_R to 4 z_R | 1.0e-06 |
| 50 cm | `focus` on 512², then `free` | complex128 | z from −4 z_R to 4 z_R | 1.1e-06 |
| 50 cm | `focus_fft` | complex128 | z = 0 z_R | 1.5e-08 |
| 50 cm | `focus_fft` | complex128 | z = 1 z_R | 1.3e-08 |
| 30 cm | `focus` | complex128 | z from −4 z_R to 4 z_R | 1.5e-08 |
| 30 cm | `focus` | complex64 | z from −4 z_R to 4 z_R | 3.1e-06 |
| 30 cm | `focus` on 512², then `free` | complex128 | z from −4 z_R to 4 z_R | 1.2e-06 |
| 30 cm | `focus_fft` | complex128 | z = 0 z_R | 1.5e-08 |
| 30 cm | `focus_fft` | complex128 | z = 1 z_R | 1.3e-08 |


### Run time

The same 1920 × 1200 SLM. Best of 5 runs (2 for the scan) on 13th Gen Intel(R) Core(TM) i7-13700, 24 logical cores.

| case | complex128 [ms] | complex64 [ms] |
|---|---|---|
| `focus`, window 128² | 12 | 5 |
| `focus`, window 256² | 25 | 10 |
| `focus`, window 512² | 60 | 30 |
| `focus`, window 256², 101 planes | 3684 | 1823 |
| `focus` on 256², then `free`, 101 planes | 260 | 171 |
| `focus_fft`, pad 2 (3840 × 2400) | 107 | 50 |
| `free`, 1920 × 1200 | 79 | 41 |

