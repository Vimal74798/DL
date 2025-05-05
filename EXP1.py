import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y = np.where(y <= 0, -1, 1)
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                if y_predicted != y[idx]:
                    self.weights += self.learning_rate * y[idx] * x_i
                    self.bias += self.learning_rate * y[idx]
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
if __name__ == "__main__":
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 0],
        [0, 1],
        [3, 1]
    ])
    y = np.array([1, 1, 1, -1, -1, -1])  
    perceptron = Perceptron(learning_rate=0.1, n_iter=10)
    perceptron.fit(X, y)
    predictions = perceptron.predict(X)
    print("Predicted labels:", predictions)
    print("Actual labels:   ", y)
OUTPUT:
Predicted labels: [ 1.  1.  1. -1. -1. -1.]
Actual labels:    [ 1  1  1 -1 -1 -1].
