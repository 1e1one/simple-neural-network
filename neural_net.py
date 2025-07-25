import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights1 = 2 * np.random.random((2, 4)) - 1
        self.weights2 = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y, iterations):
        for i in range(iterations):
            # Forward pass
            hidden = self.sigmoid(np.dot(X, self.weights1))
            output = self.sigmoid(np.dot(hidden, self.weights2))

            error = y - output
            output_delta = error * self.sigmoid_derivative(output)
            hidden_error = output_delta.dot(self.weights2.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden)

            self.weights2 += hidden.T.dot(output_delta)
            self.weights1 += X.T.dot(hidden_delta)

    def predict(self, X):
        hidden = self.sigmoid(np.dot(X, self.weights1))
        return self.sigmoid(np.dot(hidden, self.weights2))

if __name__ == "__main__":
    nn = NeuralNetwork()

    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])

    nn.train(X, y, 10000)
    print("Predictions:")
    print(nn.predict(X))
