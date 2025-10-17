import numpy as np

class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits, y_true, lr=0.01, steps=500):
        T = self.T
        for _ in range(steps):
            p = 1/(1+np.exp(-logits/T))
            grad = np.mean((p - y_true) * (logits/(T**2)))
            T -= lr*grad
            T = max(T, 1e-3)
        self.T = T
        return self

    def transform(self, logits):
        return 1/(1+np.exp(-logits/self.T))
