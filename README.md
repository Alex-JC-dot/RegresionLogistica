# Regresión Logística — Predictor de Fallos de Servidor

Modelo de regresión logística multiclase implementado **desde cero** (solo NumPy) para diagnosticar el tipo de fallo en un servidor. Usa la estrategia One-vs-All con 4 clasificadores y una interfaz gráfica con CustomTkinter.

## Características del proyecto
- Dataset simulado de 1,280 registros con ruido gaussiano
- 4 clases de salida: Fallo CPU, Fallo RAM, Fallo Red, Fallo Disco
- 6 variables de entrada: `cpu_uso`, `ram_uso`, `latencia_ms`, `temperatura_c`, `procesos`, `disco_uso`
- Visualización PCA 2D con fronteras de decisión embebida en la UI
- Modelo serializado automáticamente en `modelo.pkl` tras el entrenamiento

---

## Requisitos

- Python **3.10 o superior** (probado en 3.13)
- pip

---

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/alexcalatayud9/MAT205_IngSisCalatayudAlex.git
cd MAT205_IngSisCalatayudAlex
```

### 2. (Opcional) Crear un entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install numpy pandas matplotlib customtkinter
```

> Las librerías `os`, `pickle` y `threading` son parte de la librería estándar de Python — **no requieren instalación**.

### 4. Correr el script

```bash
python regresionLogistica.py
```

Al ejecutarlo:
1. Se abre una ventana de carga mientras se entrena el modelo (~3 segundos).
2. Aparece la UI principal con 6 campos de entrada.
3. Ingresa los valores del servidor y presiona **DIAGNOSTICAR**.
4. Se muestra la clase predicha, la confianza y el punto proyectado en la gráfica PCA.

---

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| `regresionLogistica.py` | Código principal (modelo + UI) |
| `server_failures.data` | Dataset CSV sin cabecera (1,280 filas, 7 columnas) |
| `modelo.pkl` | Modelo serializado (se genera automáticamente al correr el script) |
| `guia_regresionLogistica.txt` | Guía técnica detallada de la arquitectura y el modelo |

---

## Dependencias detalladas

| Librería | Versión mínima recomendada | Uso |
|---|---|---|
| `numpy` | 1.24 | Álgebra lineal, SVD, operaciones vectoriales |
| `pandas` | 1.5 | Carga del dataset CSV |
| `matplotlib` | 3.6 | Gráfica PCA embebida en la UI |
| `customtkinter` | 5.0 | Interfaz gráfica con tema dark |

Para instalar versiones específicas:

```bash
pip install numpy>=1.24 pandas>=1.5 matplotlib>=3.6 customtkinter>=5.0
```

---

## Guía técnica

Para detalles completos del modelo (función sigmoide, descenso de gradiente, PCA, guardado del modelo, flujo de ejecución), ver [`guia_regresionLogistica.txt`](guia_regresionLogistica.txt).
