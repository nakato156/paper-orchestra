# Idea

Queremos reducir el coste de inferencia en modelos de difusion para super-resolucion de
imagenes satelitales. La hipotesis es que la mayoria de los pasos de denoising aportan poco
en regiones de baja frecuencia (oceano, desierto), asi que proponemos un planificador que
asigna un numero de pasos distinto por parche segun su energia espectral.

Lo llamamos SpectralSkip. El objetivo es mantener la calidad perceptual bajando el computo
entre un 40 y un 60 por ciento.
