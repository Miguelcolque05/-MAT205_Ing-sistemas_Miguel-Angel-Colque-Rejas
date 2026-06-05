import numpy as np
import pandas as pd
from model import RegresionLogisticaProxy

# 1. Cargar y segmentar dataset multivariable
try:
    data = pd.read_csv('servidores_telemetria.csv')
except FileNotFoundError:
    print("Error: No se encontró el dataset. Ejecuta primero la opción 1.")
    exit()

X_raw = data[['uso_cpu', 'temperatura', 'peticiones_servidor', 'consumo_energetico']].values
y = data['caida'].values

# INGENIERÍA DE DATOS: Escalado Min-Max manual para estabilizar los gradientes
X_min = X_raw.min(axis=0)
X_max = X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)

# División 80% Entrenamiento y 20% Prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Entrenar el modelo registrando el aprendizaje en consola
print("=" * 60)
print("       MONITOR DE APRENDIZAJE DEL MODELO (LOG LOSS)       ")
print("=" * 60)

modelo = RegresionLogisticaProxy(learning_rate=0.2, iterations=10000)
num_samples = X_train.shape[0]
modelo.weights = np.zeros(X_train.shape[1])
modelo.bias = 0.0

for i in range(modelo.iterations):
    linear_model = np.dot(X_train, modelo.weights) + modelo.bias
    predictions = modelo._sigmoid(linear_model)
    
    dw = (1 / num_samples) * np.dot(X_train.T, (predictions - y_train))
    db = (1 / num_samples) * np.sum(predictions - y_train)
    
    modelo.weights -= modelo.lr * dw
    modelo.bias -= modelo.lr * db
    
    if i % 2000 == 0 or i == modelo.iterations - 1:
        eps = 1e-15
        predictions_clamped = np.clip(predictions, eps, 1 - eps)
        loss = -np.mean(y_train * np.log(predictions_clamped) + (1 - y_train) * np.log(1 - predictions_clamped))
        preds_train = np.array([1 if p >= 0.80 else 0 for p in predictions])
        acc_intermedia = np.mean(preds_train == y_train) * 100
        print(f"Iteración {i:5d} -> Pérdida (Log Loss): {loss:.5f} | Eficacia de Aprendizaje: {acc_intermedia:.2f}%")

print("=" * 60)
print("¡Optimización del Gradiente Completada exitosamente!")

# 3. MÓDULO DE AUDITORÍA: Matriz de Confusión desde Cero
predicciones_test = modelo.predict(X_test, threshold=0.80)

VP = np.sum((predicciones_test == 1) & (y_test == 1))
FP = np.sum((predicciones_test == 1) & (y_test == 0))
VN = np.sum((predicciones_test == 0) & (y_test == 0))
FN = np.sum((predicciones_test == 0) & (y_test == 1))

eps = 1e-10
precision = VP / (VP + FP + eps)
recall = VP / (VP + FN + eps)
f1_score = 2 * (precision * recall) / (precision + recall + eps)
exactitud = np.mean(predicciones_test == y_test) * 100

print(f"\n" + "="*60)
print("     MATRIZ DE CONFUSIÓN Y AUDITORÍA DE CALIDAD (SysOps)    ")
print("="*60)
print(f" [ Verdaderos Negativos (OK): {VN:3d} ]   [ Falsos Positivos (Falsa Alerta): {FP:3d} ]")
print(f" [ Falsos Negativos (No vio): {FN:3d} ]   [ Verdaderos Positivos (Alerta):   {VP:3d} ]")
print("-"*60)
print(f" -> Precisión del Sistema:      {precision * 100:.2f}%")
print(f" -> Exhaustividad (Recall):    {recall * 100:.2f}% (¡Crítica para fallos!)")
print(f" -> F1-Score Operacional:       {f1_score:.4f}")
print(f" -> Exactitud Global (Accuracy):{exactitud:.2f}%")
print("=" * 60)

# 4. PERSISTENCIA DE PARÁMETROS: Guardar los límites máximos, mínimos, pesos y sesgo
with open('pesos_modelo.txt', 'w') as f:
    f.write(f"{modelo.bias}\n")
    for w in modelo.weights:
        f.write(f"{w}\n")
    for mn in X_min:
        f.write(f"{mn}\n")
    for mx in X_max:
        f.write(f"{mx}\n")
print("[SISTEMA] Parámetros de escalado y pesos guardados en 'pesos_modelo.txt'\n")