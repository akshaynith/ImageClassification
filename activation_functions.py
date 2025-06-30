import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # For numerical stability
    return e_x / e_x.sum(axis=0)

# Generate input values
x = np.linspace(-10, 10, 100)

# Apply each activation
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_softmax = softmax(x)  # Works best with vector

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, y_sigmoid)
plt.title("Sigmoid")
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, y_tanh)
plt.title("Tanh")
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, y_relu)
plt.title("ReLU")
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, y_leaky_relu)
plt.title("Leaky ReLU")
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, y_softmax)
plt.title("Softmax (applied to vector)")
plt.grid(True)

plt.tight_layout()
plt.show()
