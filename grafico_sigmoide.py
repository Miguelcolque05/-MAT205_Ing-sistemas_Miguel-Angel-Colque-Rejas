import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import RegresionLogisticaProxy

def graficar_curva_sigmoide_real():
    # 1. Cargar dataset y parámetros optimizados
    try:
        data = pd.read_csv('servidores_telemetria.csv')
        with open('pesos_modelo.txt', 'r') as f:
            lineas = f.read().splitlines()
    except FileNotFoundError:
        print("Error: Asegúrate de haber ejecutado las opciones 1 y 2 del menú primero.")
        return

    # Extraer coeficientes del modelo
    b = float(lineas[0])
    w = np.array([float(x) for x in lineas[1:5]])
    X_min = np.array([float(x) for x in lineas[5:9]])
    X_max = np.array([float(x) for x in lineas[9:13]])

    # Preparar y normalizar los datos de prueba (20% test set para evaluar)
    X_raw = data[['uso_cpu', 'temperatura', 'peticiones_servidor', 'consumo_energetico']].values
    y_real = data['caida'].values
    X_norm = (X_raw - X_min) / (X_max - X_min + 1e-10)
    
    split = int(0.8 * len(X_norm))
    X_test = X_norm[split:]
    y_test = y_real[split:]

    # 2. Operaciones Matemáticas: Calcular Z (Combinación Lineal) y Probabilidades
    z_test = np.dot(X_test, w) + b
    
    modelo = RegresionLogisticaProxy()
    probabilidades = modelo._sigmoid(z_test)
    
    # Clasificación final con el umbral del 80% (0.80)
    umbral = 0.80
    predicciones = np.array([1 if p >= umbral else 0 for p in probabilidades])

    # 3. Determinar aciertos y errores (Misclassifications como en tu esquema)
    # Correctos = Verde Seguro / Rojo Caída bien predicha
    # Errores = Puntos amarillos o naranjas de mala clasificación
    colores_puntos = []
    for i in range(len(y_test)):
        if predicciones[i] == y_test[i]:
            colores_puntos.append('#2e7d32' if y_test[i] == 0 else '#c62828') # Verde OK, Rojo Caída
        else:
            colores_puntos.append('#ff9100') # Naranja: Error de Clasificación

    # 4. Construcción del Lienzo de la Sigmoide
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))

    # Dibujar la curva sigmoide teórica continua de fondo
    z_curva = np.linspace(z_test.min() - 2, z_test.max() + 2, 200)
    p_curva = modelo._sigmoid(z_curva)
    plt.plot(z_curva, p_curva, color='#1f538d', linewidth=3, label='Curva Logística Teórica $\sigma(z)$', zorder=1)

    # Dibujar los puntos reales evaluados sobre la curva (Proyección del modelo)
    plt.scatter(z_test, probabilidades, c=colores_puntos, s=50, alpha=0.75, edgecolors='none', label='Muestras de Servidores (Test Set)', zorder=2)

    # Línea horizontal del Umbral Crítico (80%)
    plt.axhline(y=umbral, color='#333333', linestyle='--', linewidth=1.5, label=f'Umbral de Alerta ({int(umbral*100)}%)')

    # Sombreado de Zonas (Clase Negativa vs Clase Positiva)
    plt.fill_between(z_curva, 0, umbral, where=(p_curva <= umbral), color='#2e7d32', alpha=0.05)
    plt.fill_between(z_curva, umbral, 1, where=(p_curva >= umbral), color='#c62828', alpha=0.05)

    # 5. Estética y Anotaciones Académicas
    plt.title('Mapeo de Telemetría sobre la Función Sigmoide (Regresión Logística)', fontsize=13, weight='bold', pad=15)
    plt.xlabel('Combinación Lineal Cruda del Hardware ($z = Xw + b$)', fontsize=11)
    plt.ylabel('Probabilidad Calculada de Caída ($\hat{y}$)', fontsize=11)
    plt.ylim(-0.05, 1.05)
    
    # Leyenda personalizada explicativa para la defensa
    elementos_leyenda = [
        plt.Line2D([0], [0], color='#1f538d', lw=3, label='Curva Sigmoide $\sigma(z)$'),
        plt.Line2D([0], [0], color='#333333', linestyle='--', lw=1.5, label='Umbral SAT (80%)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2e7d32', markersize=8, label='Predicción Correcta: Estable (0)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#c62828', markersize=8, label='Predicción Correcta: Caída (1)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9100', markersize=8, label='Error de Clasificación (Misclassification)')
    ]
    plt.legend(handles=elementos_leyenda, loc='upper left', frameon=True, facecolor='white')
    
    plt.tight_layout()
    plt.savefig('proyeccion_sigmoide_proxy.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    graficar_curva_sigmoide_real()