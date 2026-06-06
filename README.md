# Regresión Logística — Predictor de Fallos de Servidor

<<<<<<< HEAD
Modelo de regresión logística multiclase implementado **desde cero** (solo NumPy) para diagnosticar el tipo de fallo en un servidor. Usa la estrategia One-vs-All con 4 clasificadores y una interfaz gráfica con CustomTkinter.
=======
  regresionLogistica.py   →  Archivo principal. Contiene el
                             entrenamiento y la interfaz gráfica.

  server_failures.data    →  Dataset de 1,280 registros con 4 clases balanceadas (320 c/u).
                             Formato CSV sin cabecera.

  model_servidor.pkl      →  Modelo guardado (referencia).No es requerido para ejecutar     regresionLogistica.py


## Requisitos y Dependencias 

- Python 3.13 o superior
- numpy →  Cálculo matricial y entrenamiento
- pandas   →  Carga y procesamiento del dataset
- matplotlib →  Gráfica del espacio de decisión (PCA 2D)
- customtkinter → Interfaz gráfica moderna

Para instalar todas de una vez, abrir una terminal y ejecutar:

      pip install numpy pandas matplotlib customtkinter

Si se tiene más de una versión de Python, usar:

      py -3.13 -m pip install numpy pandas matplotlib customtkinter

# Como correr la aplicación 

OPCIÓN A — Desde terminal ()
  1. Abrir cmd o PowerShell.
  2. Navegar a la carpeta del proyecto:

         cd "D:\Archivo_donde_esta_regresionLogistica

  3. Ejecutar:

         python regresionLogistica.py

     O si hay múltiples versiones de Python:

         py -3.13 regresionLogistica.py

OPCIÓN B — Desde un IDE (VS Code, PyCharm, etc):
  1. Abrir la carpeta del proyecto en el IDE.
  2. Seleccionar el intérprete Python 3.13.
  3. Abrir regresionLogistica.py y presionar Run.

>>>>>>> 4ee7d910875bbecca769a0807593669cfd2c96bf

## Características del proyecto
- Dataset simulado de 1,280 registros con ruido gaussiano
- 4 clases de salida: Fallo CPU, Fallo RAM, Fallo Red, Fallo Disco
- 6 variables de entrada: `cpu_uso`, `ram_uso`, `latencia_ms`, `temperatura_c`, `procesos`, `disco_uso`
- Visualización PCA 2D con fronteras de decisión embebida en la UI
- Modelo serializado automáticamente en `modelo.pkl` tras el entrenamiento

---

<<<<<<< HEAD
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
=======
## Planteamiento del problema 

En la administración de servidores, cuando ocurre un fallo, el equipo técnico necesita saber rápidamente qué tipo de fallo es para tomar la acción correcta. No es lo mismo un problema de CPU (que requiere optimizar procesos o actualizar hardware) que un fallo de red (que requiere revisar conectividad) o un fallo de disco (que requiere reemplazar almacenamiento).
El problema es: dado un conjunto de 6 métricas continuas que un servidor reporta en tiempo real (uso de CPU, uso de RAM, latencia de red, temperatura del procesador, cantidad de procesos activos y uso de disco), clasificar automáticamente el tipo de fallo entre 4 categorías posibles: sobrecarga de CPU, fallo de memoria RAM, fallo de red o fallo de disco.


## Sustento matemático 
La regresión logística La Regresión Logística es un algoritmo de aprendizaje supervisado utilizado para predecir una variable categórica a partir de variables continuas. A pesar de su nombre, no es un método de regresión sino de clasificación, ya que su salida es una probabilidad entre 0 y 1 que se asigna a una clase

Lo que pasaba con la regresión lineal es que usabas el error cuadrático medio, la minimizas (buscar el mínimo de una función), encuentras los coeficientes y la recta poder hacer la simulación. Sin embargo cuando se trata de una respuesta binario (si o no) es un poco complicado, nos puede soltar valores que no tengan sentido para la clasificación, mientras la regresión logística usa una combinación lineal de los datos de entrada para la predicción, pero comprime su resultado para que se ajuste a 0 y 1.

<p align = "center">
  <img width="256" height="47" alt="Captura de pantalla 2026-06-05 011618"                 src="https://github.com/user-attachments/assets/f697ea64-0cb3-4ee6-9889-908b8018ee07" />
  <br>
</p>

|  Regresión Lineal |  Regresión Logística  |
| :---: | :---: |
| <img width="426" height="330" alt="Captura de pantalla 2026-06-05 013353" src="https://github.com/user-attachments/assets/e262f70f-1e99-432a-800b-f5dee972e0be" />| <img width="400" alt="Regresión Logística" src="https://github.com/user-attachments/assets/c4f4cab1-10e0-49b6-a2fd-b02bf312566b" /> |
| *La recta predice valores fuera de 0 y 1* | *La curva se adapta perfectamente a los límites binarios.* |

---


### Sigmoide 
Lo comprime con usando la función sigmoide una curva de forma suave, que asigna a un valor un número entre 0 y 1.

<p align = "center">
  <img width="237" height="113" alt="Captura de pantalla 2026-06-05 012901" src="https://github.com/user-attachments/assets/23086297-9d51-40f7-b552-dceb3c7fd9e0" />
  <br>
  <i>Sigmoide</i>
</p>

Se interpreta el valor como la probabilidad de un resultado por ejemplo:
* 0 = 0 % probabilidades de que si suceda
* 0.27 = 27% probabilidades de que si suceda
* 1 = 100% probabilidades de que si suceda

### Función de costo - Log Loss

¿Por qué no usar sigmoide?  Porque hace que la función de error quede con muchos mínimos locales y el descenso de gradiente se atasca. Necesitas una función más "amigable" matemáticamente.
Normalmente cuando haces una predicción y fallas mucho depende del porcentaje de tu predicción, es decir que mientras más pequeño sea tu porcentaje menos riesgo habrá.

<p align = "center">
 <img width="371" height="66" alt="Captura de pantalla 2026-06-05 044414" src="https://github.com/user-attachments/assets/59bdb00c-2210-415a-9026-8f3528ae3da8" />
  <br>
  <i> Función de logarítmica</i>
</p>

La función logarítmica tiene una propiedad clave: cuando la predicción 
es correcta y cercana a 1, el costo es casi 0. Pero cuando la predicción 
es muy incorrecta (cercana a 0 cuando debería ser 1), el costo crece 
exponencialmente. Esto obliga al modelo a evitar predicciones confiadas 
pero equivocadas.

| Predicción (p) | Clase real = 1: -log(p) | Interpretación        |
|----------------|-------------------------|-----------------------|
| 0.99           | 0.01                    | Casi perfecto         |
| 0.80           | 0.22                    | Bien pero con duda    |
| 0.50           | 0.69                    | No sabe               |
| 0.10           | 2.30                    | Muy equivocado        |
| 0.01           | 4.60                    | Desastrosamente mal   |

Es mucho mejor la penalización y no habrá bache, como en una simple división donde la forma de penalización no siempre tiene problemas.

pensemos en el problema planteado (de forma binaria) *diagnóstico de servidores* veamos los datos de entrenamiento que tenemos, ahora para cualquier dato de los coeficientes y el termino sesgo de nuestro modelo, podemos calcular la probabilidad, observamos los datos.
* Servidor respondiendo = p
* Servidor fallando = 1- p (abusando un poco de la notación)

<p align ="center">
  <img width="414" height="122" alt="Captura de pantalla 2026-06-04 231921" src="https://github.com/user-attachments/assets/983c355a-9fa0-45be-a66f-e60e94d6ef49" />
  <br>
  <br>
  <img width="479" height="68" alt="Captura de pantalla 2026-06-05 014910"   src="https://github.com/user-attachments/assets/d4d8f161-f671-47fb-bedd-46d38242880a" />
  <br>
  <i>Función de costo</i>
</p>

### Descenso de gradiente
para ajusta u "optimizar" los pesos usaremos el descenso de la gradiente para ajustar los pesos

<p align ="center">
<img width="194" height="72" alt="Captura de pantalla 2026-06-05 015115" src="https://github.com/user-attachments/assets/1de17499-9b87-44a3-8cee-b2f096cfca04" />

  <br>
  <i>Descenso de gradiente</i>
</p>

## One vs All
Normalmente con la regresión logística basta, ¿qué pasa si tenemos más de dos respuestas?  Por ejemplo en nuestro caso: queremos ver si el servidor está bien, la respuesta binaria seria si o no, pero ahora podemos definir que tipo de falla en concreto o si simplemente no falla. Ahí entra el one vs all.
La estrategia consiste en entrenar un clasificador logístico independiente por cada clase. Cada clasificador aprende a responder una sola pregunta: ¿pertenece este dato a mi clase o no

*	Clasificador 1: ¿problema de CPU vs el resto? 
* Clasificador 2: ¿problema de Red vs el resto? 
*	Clasificador 3: ¿problema de RAM  vs el resto?
*	Clasificador 4: ¿problema de disco  vs el resto?


|  Todas las clases | 
| :---: | 
| <img width="302" height="239" alt="Captura de pantalla 2026-06-05 005424" src="https://github.com/user-attachments/assets/f4a584db-dd33-4b54-9304-ff0196e67de4" />
| *Cada uno sigue el mismo proceso que ya vimos: calcula z, aplica la sigmoide, obtiene una probabilidad p y minimiza su propio Log-Loss con descenso de gradiente.* |

---

### Argmax
Entre todas las probabilidades vemos quien es el que tiene más valor o en otras palabras cual es el más seguro que suceda

<p align ="center">
<img width="252" height="69" alt="Captura de pantalla 2026-06-05 020150" src="https://github.com/user-attachments/assets/01bb0041-786e-4f9d-bee6-346fee2c2a65" />
</p>

En nuestro caso, el modelo recibe 6 métricas continuas de un servidor:
cpu_uso, ram_uso, latencia_ms, temperatura_c, procesos y disco_uso.

Cada clasificador aprende qué variables importan más para su clase. 
Por ejemplo, el clasificador de CPU aprenderá pesos altos para 
cpu_uso y temperatura_c, mientras que el de Red aprenderá un peso 
alto para latencia_ms.

Ejemplo de predicción:
| Entrada: CPU=95%, RAM=40%, Lat=12ms, Temp=82°C, Proc=110, Disco=45% |

| Clasificador | Probabilidad |
|-------------|--------------|
| CPU         | 99.6%  ← ganador |
| RAM         | 0.2%         |
| Red         | 0.1%         |
| Disco       | 0.1%         |
>>>>>>> 4ee7d910875bbecca769a0807593669cfd2c96bf
