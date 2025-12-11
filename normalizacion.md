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
x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

Esto garantiza que **todas las entradas queden entre 0 y 1**, permitiendo:

- Comparar variables distintas en igualdad de condiciones.  
- Entrenar modelos (redes neuronales, regresión, clustering) sin que un atributo domine a los demás.  
- Graficar valores heterogéneos sin perder su forma relativa.

Cada columna tiene **su propio mínimo y máximo**, por lo que cada variable se normaliza independientemente.

```python
import numpy as np

# Matriz de datos:
# Columna 0 = Edad
# Columna 1 = Ingresos
xn = np.array([
    [20, 6000],
    [35, 40000],
    [29, 12000],
    [50, 35000]
])
print("Datos originales:\n", xn)

# 1. Obtener el valor mínimo de cada columna (edad mínima, ingreso mínimo)
min_vals = xn.min(axis=0)
print("\nMínimos por columna (edad, ingresos):\n", min_vals)

# 2. Obtener el valor máximo de cada columna (edad máxima, ingreso máximo)
max_vals = xn.max(axis=0)
print("\nMáximos por columna (edad, ingresos):\n", max_vals)

# 3. Normalizar usando la fórmula:
#    x_normalizado = (x - min) / (max - min)
xn_norm = (xn - min_vals) / (max_vals - min_vals)

print("\nDatos normalizados entre 0 y 1:\n", xn_norm)
```
## Resumen de Normalización de Múltiples Entradas

### Datos Originales | Mínimos | Máximos | Datos Normalizados
| Edad | Ingresos | Min Edad | Min Ingresos | Max Edad | Max Ingresos | Edad (norm) | Ingresos (norm) |
|------|----------|----------|--------------|----------|--------------|--------------|------------------|
| 20   | 6000     | 20       | 6000         | 50       | 40000        | 0.00         | 0.00             |
| 35   | 40000    | 20       | 6000         | 50       | 40000        | 0.50         | 1.00             |
| 29   | 12000    | 20       | 6000         | 50       | 40000        | 0.30         | 0.1875           |
| 50   | 35000    | 20       | 6000         | 50       | 40000        | 1.00         | 0.875            |


<img width="453" height="385" alt="image" src="https://github.com/user-attachments/assets/a2351f3c-8e37-4c96-aee4-62ba82492074" />

---
### ¿Qué hace la normalización dinámica?

La normalización dinámica ajusta cada característica usando **el mínimo y el máximo reales del conjunto de datos**.  
Así, cada columna se transforma para quedar entre **0 y 1**, sin importar sus unidades originales.

Esto permite:
- Comparar variables con escalas muy distintas (por ejemplo, edad vs ingresos).
- Evitar que valores grandes dominen a valores pequeños.
- Adaptarse automáticamente a cualquier conjunto de datos, porque siempre calcula nuevos mínimos y máximos.

```python
import numpy as np

# Datos: [Edad, Ingreso]
xn = np.array([
    [20, 6000],
    [35, 40000],
    [29, 12000],
    [50, 35000]
])

print("Datos originales:\n", xn)

# Normalización con rangos dinámicos (del dataset)
min_dyn = xn.min(axis=0)
max_dyn = xn.max(axis=0)

print("\nMínimos dinámicos:", min_dyn)
print("Máximos dinámicos:", max_dyn)

# Normalización
xn_norm_dynamic = (xn - min_dyn) / (max_dyn - min_dyn)

print("\nNormalización dinámica (0 a 1):\n", xn_norm_dynamic)
```
```txt
Datos originales:
 [[   20  6000]
 [   35 40000]
 [   29 12000]
 [   50 35000]]

Mínimos dinámicos:
 [   20  6000]

Máximos dinámicos:
 [   50 40000]

Normalización dinámica (0 a 1):
 [[0.000 0.000]
 [0.500 1.000]
 [0.300 0.176]
 [1.000 0.853]]
```

---
### ¿Qué es la normalización fija?

La normalización fija consiste en escalar los datos usando **rangos definidos por el usuario**, no por los valores reales del conjunto de datos.  
Es decir, tú estableces manualmente el mínimo y el máximo permitidos para cada característica (por ejemplo, edad de 20 a 65, ingresos de 5000 a 50000).

### ¿Qué implica?

- Todos los datos se llevan a una escala entre 0 y 1 **siempre que estén dentro del rango fijado**.
- Si un dato cae **afuera del rango**, su normalización puede ser:
  - **menor que 0** (si es menor al mínimo)
  - **mayor que 1** (si supera el máximo)

### ¿Para qué sirve?

- Útil cuando trabajas con sensores, modelos o sistemas que **tienen un rango conocido y fijo**.
- Mantiene una escala constante sin importar cómo cambien los datos.
- Permite comparar nuevas entradas contra un estándar establecido.

### Diferencia clave vs normalización dinámica

- **Normalización dinámica:** el rango se calcula usando tus propios datos.
- **Normalización fija:** el rango lo defines tú y no cambia, aunque tus datos sí cambien.

```python
import numpy as np

# Datos: [Edad, Ingreso]
xn = np.array([
    [20, 6000],
    [35, 40000],
    [29, 12000],
    [50, 35000]
])

print("Datos originales:\n", xn)

# Rangos fijos establecidos
range_age = [20, 65]          # Mín y máx posibles de edad
range_income = [5000, 50000]  # Mín y máx posibles de ingresos

min_fixed = np.array([range_age[0], range_income[0]])
max_fixed = np.array([range_age[1], range_income[1]])

print("\nMínimos fijos:", min_fixed)
print("Máximos fijos:", max_fixed)

# Normalización con valores fijos
xn_norm_fixed = (xn - min_fixed) / (max_fixed - min_fixed)

print("\nNormalización con rangos fijos:\n", xn_norm_fixed)
```
```txt
Datos originales:
 [[   20  6000]
 [   35 40000]
 [   29 12000]
 [   50 35000]]

Mínimos fijos:
 [   20  5000]

Máximos fijos:
 [   65 50000]

Normalización con rangos fijos:
 [[0.000 0.022]
 [0.333 0.778]
 [0.200 0.156]
 [0.667 0.667]]
```

