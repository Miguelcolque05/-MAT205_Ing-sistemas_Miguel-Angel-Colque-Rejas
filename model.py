import numpy as np

class RegresionLogisticaProxy:
    """
    Clase que implementa el algoritmo de Regresión Logística desde cero
    para la predicción de fallos en servidores proxy.
    """
    def __init__(self, learning_rate=0.001, iterations=5000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """Calcula la función de activación sigmoide."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Entrena el modelo utilizando optimización por Gradiente Descendiente."""
        num_samples, num_features = X.shape
        
        # Inicialización de parámetros en cero
        self.weights = np.zeros(num_features)
        self.bias = 0.0

        # Bucle de optimización
        for _ in range(self.iterations):
            # Combinación lineal: z = Xw + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Mapeo probabilístico: y_hat = sigma(z)
            predictions = self._sigmoid(linear_model)

            # Cálculo de gradientes (derivadas parciales de Log Loss)
            dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
            db = (1 / num_samples) * np.sum(predictions - y)

            # Actualización simultánea de parámetros
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """Devuelve la probabilidad continua calculada (entre 0 y 1)."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.80):
        """Clasifica la salida en variables discretas (0 o 1) según el umbral."""
        probabilities = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in probabilities])