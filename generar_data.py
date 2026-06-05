import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad matemática
np.random.seed(42)
n_muestras = 1000

# 1. Simular las 4 métricas de hardware y red
uso_cpu = np.random.uniform(10.0, 100.0, n_muestras)
temperatura = np.random.uniform(40.0, 95.0, n_muestras)
peticiones = np.random.uniform(100, 5000, n_muestras)       # Req/seg
consumo_watts = np.random.uniform(50.0, 250.0, n_muestras)   # Consumo energético

# 2. Regla lógica combinada con ruido Gaussiano para determinar la caída
ruido = np.random.normal(0, 8, n_muestras)
condicion = (0.25 * uso_cpu) + (0.35 * temperatura) + (0.01 * peticiones) + (0.10 * consumo_watts) + ruido

# Umbral crítico de colapso
caida = (condicion > 95).astype(int)

# 3. Construir el DataFrame expandido
df = pd.DataFrame({
    'uso_cpu': np.round(uso_cpu, 1),
    'temperatura': np.round(temperatura, 1),
    'peticiones_servidor': np.round(peticiones, 0),
    'consumo_energetico': np.round(consumo_watts, 1),
    'caida': caida
})

df.to_csv('servidores_telemetria.csv', index=False)
print("¡Dataset expandido (4 variables) creado con éxito directamente en la raíz!")
print(f"Estadísticas -> Registros Normales (0): {sum(caida == 0)} | Servidores Caídos (1): {sum(caida == 1)}")