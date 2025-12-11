# Creacion de las funciones de activacion

**1. Sigmoide (σ(x))**  
- Rango: (0, 1)  
- Uso típico: probabilidades  
- Ventajas: salida suave  
- Desventajas: saturación y gradiente pequeño
```python
# Gráfica de la función sigmoide

import numpy as np
import matplotlib.pyplot as plt

# Generación de valores en el eje x
# np.linspace(inicio, fin, cantidad_de_puntos)
x = np.linspace(-6, 6, 100)

# Cálculo de la función sigmoide
# Fórmula: σ(x) = 1 / (1 + e^(-x))
sigmoid = 1 / (1 + np.exp(-x))

# Graficar la funcións
plt.plot(x, sigmoid)
plt.xlabel('x')                         # Etiqueta del eje x
plt.ylabel('Función sigmoide')          # Etiqueta del eje y
plt.title('Función Sigmoide')           # Título de la gráfica
plt.grid(True)                          # Cuadrícula para facilitar lectura

# Mostrar la gráfica
plt.show()

```
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/39c3fa9d-7517-4134-8822-dc804fdc0c93" />

**2. Tanh (tanh(x))**  
- Rango: (-1, 1)  
- Uso típico: capas ocultas centradas en cero  
- Ventajas: mejor que sigmoide para redes profundas  
- Desventajas: también sufre saturación
# Gráfica de la función ReLU
**3. ReLU (max(0, x))**  
- Rango: [0, ∞)  
- Uso típico: redes profundas modernas  
- Ventajas: entrenamiento rápido, evita saturación para x>0  
- Desventajas: neuronas muertas (x<0 siempre da 0)
```python
import numpy as np
import matplotlib.pyplot as plt

# Generación de valores en el eje x
x = np.linspace(-6, 6, 100)

# Cálculo de la función ReLU
# Fórmula: ReLU(x) = max(0, x)
relu = np.maximum(0, x)

# Graficar la función ReLU
plt.plot(x, relu)
plt.xlabel('x')                      # Etiqueta del eje x
plt.ylabel('Función ReLU')           # Etiqueta del eje y
plt.title('Función ReLU')            # Título de la gráfica
plt.grid(True)                       # Cuadrícula para facilitar lectura

# Mostrar la gráfica
plt.show()
```
<img width="554" height="455" alt="image" src="https://github.com/user-attachments/assets/b95abc7a-d344-4277-a971-8cd3dbc3eb54" />
