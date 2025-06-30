import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))  # For numerical stability
    return e_x / e_x.sum(axis=0)

# Hypothetical input array
input_array = np.array([1.0, 2.0, 3.0, 4.0])

# Apply the softmax function
output_array = softmax(input_array)

# Print the input and output
print("Input array:", input_array)
print("Softmax output:", output_array)
print("Sum of softmax output:", np.sum(output_array))