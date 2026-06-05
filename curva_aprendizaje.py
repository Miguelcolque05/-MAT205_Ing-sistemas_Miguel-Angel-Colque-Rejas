import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import RegresionLogisticaProxy

def graficar_curva_aprendizaje():
    try:
        data = pd.read_csv('servidores_telemetria.csv')
    except FileNotFoundError:
        print("Error: Ejecuta primero la opción 1 del menú.")
        return

    X_raw = data[['uso_cpu', 'temperatura', 'peticiones_servidor', 'consumo_energetico']].values
    y = data['caida'].values
    X = (X_raw - X_raw.min(axis=0)) / (X_raw.max(axis=0) - X_raw.min(axis=0))

    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]

    modelo = RegresionLogisticaProxy(learning_rate=0.2, iterations=10000)
    num_samples = X_train.shape[0]
    modelo.weights = np.zeros(X_train.shape[1])
    modelo.bias = 0.0

    historial_iteraciones = []
    historial_perdida = []
    historial_eficacia = []

    print("[PROCESO] Analizando épocas de optimización para graficar...")
    
    for i in range(modelo.iterations):
        linear_model = np.dot(X_train, modelo.weights) + modelo.bias
        predictions = modelo._sigmoid(linear_model)
        
        dw = (1 / num_samples) * np.dot(X_train.T, (predictions - y_train))
        db = (1 / num_samples) * np.sum(predictions - y_train)
        
        modelo.weights -= modelo.lr * dw
        modelo.bias -= modelo.lr * db
        
        if i % 100 == 0 or i == modelo.iterations - 1:
            eps = 1e-15
            predictions_clamped = np.clip(predictions, eps, 1 - eps)
            loss = -np.mean(y_train * np.log(predictions_clamped) + (1 - y_train) * np.log(1 - predictions_clamped))
            preds_train = np.array([1 if p >= 0.80 else 0 for p in predictions])
            acc = np.mean(preds_train == y_train) * 100
            
            historial_iteraciones.append(i)
            historial_perdida.append(loss)
            historial_eficacia.append(acc)

    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(historial_iteraciones, historial_perdida, color='#c62828', linewidth=2.5)
    ax1.set_title('Evolución de la Función de Pérdida (Log Loss)', fontsize=12, weight='bold', pad=10)
    ax1.set_xlabel('Iteraciones (Épocas)')
    ax1.set_ylabel('Costo Matemático (Loss)')

    ax2.plot(historial_iteraciones, historial_eficacia, color='#2e7d32', linewidth=2.5)
    ax2.set_title('Eficacia del Aprendizaje (Accuracy %)', fontsize=12, weight='bold', pad=10)
    ax2.set_xlabel('Iteraciones (Épocas)')
    ax2.set_ylabel('Porcentaje de Acierto (%)')
    ax2.set_ylim(50, 105)

    plt.suptitle('TelcoBolivia S.A. - Monitoreo de Convergencia del Gradiente Descendiente', fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('curva_aprendizaje_proxy.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    graficar_curva_aprendizaje()