# Log experimental

Dataset: WorldStrat (3200 escenas), ampliacion x4. Baseline: StableSR con 50 pasos uniformes.

| Metodo | Pasos medios | PSNR | SSIM | LPIPS | GFLOPs |
|---|---|---|---|---|---|
| StableSR (baseline) | 50.0 | 28.41 | 0.812 | 0.183 | 1240 |
| Uniforme-25 | 25.0 | 27.02 | 0.771 | 0.241 | 620 |
| SpectralSkip (nuestro) | 21.3 | 28.36 | 0.809 | 0.188 | 528 |

Ablacion del umbral de energia: tau=0.1 da 24.8 pasos y LPIPS 0.185; tau=0.3 da 21.3 pasos y
LPIPS 0.188; tau=0.5 da 16.1 pasos pero LPIPS se degrada a 0.227.

Observacion: la ganancia se concentra en escenas con mas de 60 por ciento de agua. En escenas
urbanas el planificador casi no salta pasos (media 44.2), como era de esperar.

Fallo pendiente: en 7 de 3200 escenas el planificador colapsa a 4 pasos y produce artefactos
de bloque. No tenemos aun una explicacion.
