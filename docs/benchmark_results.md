### Accuracy

Gaussian beam of radius 1.2 mm on a 1920 × 1200 SLM of 8 µm pixels, 1030 nm, f = 50 cm, so w₀ = 137 µm and z_R = 5.7 cm. Largest difference over a window of ±3w₀, relative to the peak.

| SLM–lens distance | method | precision | planes | error |
|---|---|---|---|---|
| 50 cm | `focus` | complex128 | back focal plane | 1.5e-08 |
| 50 cm | `focus` | complex64 | back focal plane | 9.0e-08 |
| 50 cm | `focus`, then `free` on ±2.5 mm | complex128 | z from −4 z_R to 4 z_R | 1.1e-06 |
| 30 cm | `focus` | complex128 | back focal plane | 1.5e-08 |
| 30 cm | `focus` | complex64 | back focal plane | 1.0e-07 |
| 30 cm | `focus`, then `free` on ±2.5 mm | complex128 | z from −4 z_R to 4 z_R | 1.2e-06 |

