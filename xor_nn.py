import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def init_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

def dot_product(weights, inputs):
    return [sum(w * i for w, i in zip(row, inputs)) for row in weights]

training_inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

training_outputs = [
    [0],
    [1],
    [1],
    [0]
]

input_size = 2
hidden_size = 2
output_size = 1

weights_input_hidden = init_weights(hidden_size, input_size)
weights_hidden_output = init_weights(output_size, hidden_size)

epochs = 10000
learning_rate = 0.5

for epoch in range(epochs):
    total_error = 0
    for inputs, expected_output in zip(training_inputs, training_outputs):
        hidden_input = dot_product(weights_input_hidden, inputs)
        hidden_output = [sigmoid(h) for h in hidden_input]

        final_input = dot_product(weights_hidden_output, hidden_output)
        final_output = [sigmoid(f) for f in final_input]

        error = [expected_output[i] - final_output[i] for i in range(output_size)]
        total_error += sum(e ** 2 for e in error)

        d_output = [error[i] * sigmoid_derivative(final_output[i]) for i in range(output_size)]

        d_hidden = []
        for i in range(hidden_size):
            delta = sum(d_output[j] * weights_hidden_output[j][i] for j in range(output_size))
            d_hidden.append(delta * sigmoid_derivative(hidden_output[i]))

        for i in range(output_size):
            for j in range(hidden_size):
                weights_hidden_output[i][j] += learning_rate * d_output[i] * hidden_output[j]

        for i in range(hidden_size):
            for j in range(input_size):
                weights_input_hidden[i][j] += learning_rate * d_hidden[i] * inputs[j]

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {total_error:.4f}")

print("\n--- Test ---")
for inputs in training_inputs:
    hidden_input = dot_product(weights_input_hidden, inputs)
    hidden_output = [sigmoid(h) for h in hidden_input]

    final_input = dot_product(weights_hidden_output, hidden_output)
    final_output = [sigmoid(f) for f in final_input]

    print(f"{inputs} => {round(final_output[0], 3)}")
