
Requisitos e Instalación
1 Instalación de Dependencias
py -m pip install numpy pandas

2 Instalación de Dependencias de la Interfaz y Gráficos

py -m pip install customtkinter matplotlib seaborn

*Flujo de Ejecución y Uso

Paso 1: Generación del Dataset 
py generar_data.py

Esto creará el archivo servidores_telemetria.csv en la raíz del proyecto.
Paso 2: Entrenamiento del Modelo Numérico

py train.py

Corre el algoritmo de optimización por Gradiente Descendiente para ajustar las matrices de pesos (w) y sesgo (b) 

py train.py

Este script dividirá los datos (80/20), guardará los coeficientes optimizados y desplegará la Matriz de Confusión de validación.

Paso 3: Lanzamiento del Entorno(dos opciones de hacerlo)

Opción A (Consola Centralizada): Administra todo el flujo, gráficos y pruebas mediante el menú numérico interactivo:

py menu.py

Opción B (Interfaz Gráfica en Vivo): Lanza directamente el panel de control con sliders para realizar inferencias probabilísticas en tiempo real

py gui.py