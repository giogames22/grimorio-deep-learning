# Normalización de Datos  
### Definicion de la normalizacion

La normalización es una técnica que transforma valores de diferentes escalas a una escala común, normalmente entre 0 y 1.  
Esto facilita el análisis, la comparación y el procesamiento, especialmente en áreas como machine learning, señales y control.

La fórmula general de la normalización es:

$$
\frac{x - x_{min}}{x_{max} - x_{min}}
$$

### Factor de normalización

El factor de normalización se define como:

$$
\text{factor} = \frac{1}{x_{\max} - x_{\min}}
$$

---

# Demostracion del factor denormalizacion
```python
import numpy as np 
import matplotlib.pyplot as plt
#Rango de pixel 
pixel = np.linspace(0,255,100)
#rngo de voltajes
voltajes = np.linspace(0,5,100)

#obtencion de los valores minimos de cada uno de los parametros
pixelmin = np.min(pixel)
voltajesmin = np.min(voltajes)

#obtecion de los valores maximos de voltaje
pixelmax = np.max(pixel)
voltajesmax = np.max(voltajes)

##facoctor de la normalizacion
factor_pixel = 1 / (pixelmax - pixelmin)
factor_voltaje = 1 / (voltajesmax - voltajesmin)
#creacion de los valores normalizados
pixel_normalizado = (pixel - pixelmin) * factor_pixel
voltajes_normalizados = (voltajes - voltajesmin) * factor_voltaje

print("Factor de normalización pixel:", factor_pixel)
print("Factor de normalización voltaje:", factor_voltaje)
```
### Salida del programa

```txt
Factor de normalización pixel: 0.00392156862745098
Factor de normalización voltaje: 0.2
```
