# Creacion de las funciones de activacion

**1. Sigmoide (σ(x))**  
- Rango: (0, 1)  
- Uso típico: probabilidades  
- Ventajas: salida suave  
- Desventajas: saturación y gradiente pequeño
- 
```python
# Gráfica de la función sigmoide

import numpy as np
import matplotlib.pyplot as plt

# Generación de valores para el eje x
# (pueden ser más puntos para una curva más suave)
x = np.array([-2, -1, 0, 1, 2])

# Cálculo de la función sigmoide para cada valor de x
# Fórmula: σ(x) = 1 / (1 + e^(-x))
sigmoid = 1 / (1 + np.exp(-x))

# Configuración de la gráfica
plt.plot(x, sigmoid, marker='o')   # Línea con marcadores en cada punto
plt.xlabel('x')                    # Etiqueta del eje x
plt.ylabel('Función sigmoide')     # Etiqueta del eje y
plt.title('Gráfica de la función sigmoide')  # Título opcional
plt.grid(True)                     # Mostrar cuadrícula para mejor lectura

# Mostrar gráfica en pantalla
plt.show()
```

**2. Tanh (tanh(x))**  
- Rango: (-1, 1)  
- Uso típico: capas ocultas centradas en cero  
- Ventajas: mejor que sigmoide para redes profundas  
- Desventajas: también sufre saturación

**3. ReLU (max(0, x))**  
- Rango: [0, ∞)  
- Uso típico: redes profundas modernas  
- Ventajas: entrenamiento rápido, evita saturación para x>0  
- Desventajas: neuronas muertas (x<0 siempre da 0)
