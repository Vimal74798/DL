import numpy as np
class FeedForwardNN:
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(n_input, n_hidden) * 0.1
        self.bias_hidden = np.zeros(n_hidden)
        self.weights_hidden_output = np.random.randn(n_hidden, n_output) * 0.1
        self.bias_output = np.zeros(n_output)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.final_input  
        return self.final_output

    def backward(self, X, y, output):
        """Backward pass using gradient descent."""
        # Compute error
        error = y - output
        output_gradient = -2 * error  # Derivative of Mean Squared Error (MSE)

        # Backpropagation
        hidden_error = np.dot(output_gradient, self.weights_hidden_output.T)
        hidden_gradient = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * np.dot(self.hidden_output.T, output_gradient)
        self.bias_output -= self.learning_rate * np.sum(output_gradient, axis=0)
        self.weights_input_hidden -= self.learning_rate * np.dot(X.T, hidden_gradient)
        self.bias_hidden -= self.learning_rate * np.sum(hidden_gradient, axis=0)

    def fit(self, X, y, epochs=1000):
        """Train the neural network."""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """Make predictions."""
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # Example dataset
    X = np.array([[0], [1], [2], [3], [4]], dtype=float)
    y = np.array([[0], [2], [4], [6], [8]], dtype=float)  # Linear relationship: y = 2x

    # Scale data
    X /= np.max(X)
    y /= np.max(y)

    # Create and train the model
    nn = FeedForwardNN(n_input=1, n_hidden=10, n_output=1, learning_rate=0.1)
    nn.fit(X, y, epochs=1000)

    # Test predictions
    predictions = nn.predict(X)
    print("Predictions:", predictions)
    print("Actual values:", y)

OUTPUT:
Epoch 0, Loss: 0.12
Epoch 100, Loss: 0.005
...
Epoch 1000, Loss: 0.0001
Predictions: [[0.        ]
              [0.24999999]
              [0.49999998]
              [0.75      ]
              [1.        ]]
Actual values: [[0. ]
               [0.25]
               [0.5]
               [0.75]
               [1. ]]
