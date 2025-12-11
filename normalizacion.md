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
---
### ¿Qué pasa si no normalizamos?

Si no normalizamos, las dos variables quedan en **escalas muy diferentes** (por ejemplo, pixels de 0 a 255 y voltajes de 0 a 5).  
Al graficarlas juntas en el mismo eje:

- La señal pequeña (voltaje) se **aplana** y parece casi una línea recta.  
- No se pueden **comparar tendencias** ni la forma real de las dos señales.  
- La gráfica pierde **interpretación** porque la escala la domina la variable más grande.

Por eso se normaliza: para llevar ambas señales al mismo rango y poder visualizarlas y compararlas correctamente.
```python
# Create plot
plt.plot(pixel, label="Pixel")
plt.plot(voltajes, label="Voltajes")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.title("Comparación entre Pixels y Voltajes (sin normalizar)")
plt.grid(True)
plt.legend()

plt.show()
```
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/a897fe63-117b-4b02-b310-df1422594241" />

---
### ¿Qué pasa cuando normalizamos?

Al normalizar, ambas señales se transforman al mismo rango (0 a 1).  
Esto permite que:

- Las dos curvas puedan **verse completas** en la misma gráfica.  
- Sea posible **comparar sus formas** sin que una “aplasté” a la otra.  
- Ambas variables tengan **igual peso visual**, aunque tengan unidades distintas.

La normalización no cambia la forma de la señal; solo ajusta su escala para hacerla comparable.
En este caso al crecer de la misma manera ambas quedan superpuestas. 

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/5011e5a9-5421-4b5f-b2a7-388d40ee2414" />

---
### Normalización con múltiples entradas

Cuando trabajamos con varios parámetros al mismo tiempo (por ejemplo: edad, ingresos, voltaje, temperatura, pixeles, etc.), cada uno tiene **unidades y escalas diferentes**.  
Si no normalizamos cada entrada por separado, los valores con escalas más grandes dominan el análisis o los modelos.

Por eso, se realiza la normalización **columna por columna**, aplicando:

$$
\text{x_{norm}} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Esto garantiza que **todas las entradas queden entre 0 y 1**, permitiendo:

- Comparar variables distintas en igualdad de condiciones.  
- Entrenar modelos (redes neuronales, regresión, clustering) sin que un atributo domine a los demás.  
- Graficar valores heterogéneos sin perder su forma relativa.

Cada columna tiene **su propio mínimo y máximo**, por lo que cada variable se normaliza independientemente.

