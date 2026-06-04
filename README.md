# Regresión Logistica 

Este es un trabajo donde se entrenó un modelo para la clasificación de fallos en un servidor, usando Regresión Logística implementada desde cero (sin sklearn). No solo diagnostica si el servidor está bien o mal, sino que con la estrategia One-vs-All identifica exactamente el tipo de falla.

# Instrucciones del script


## Características del proyecto
- Dataset simulado de 1,280 registros con ruido gaussiano
- 4 variables categóricas de salida (Fallo CPU, Fallo RAM, Fallo Red, Fallo Disco)
- 6 variables continuas de entrada (cpu_uso, ram_uso, latencia_ms, temperatura_c, procesos, disco_uso)
- Interfaz gráfica con CustomTkinter para diagnóstico en tiempo real

## Requisitos
- Python 3.13 o superior
- numpy
- pandas
- customtkinter
- matplotlib

