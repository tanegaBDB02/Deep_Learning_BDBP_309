# Implement the following functions in Python from scratch.
# Do not use any library functions. You are allowed to use numpy and matplotlib.
# Generate 100 equally spaced values between -10 and 10. Call this list as  z.
# Implement the following functions and its derivative.
# Use class notes to find the expression for these functions.
# Use z as input and plot both the function outputs and its derivative outputs.
#
# Sigmoid
# Tanh
# ReLU (Rectified Linear Unit)
# Leaky ReLU
# Softmax

import matplotlib.pyplot as plt
import numpy as np
from math import exp

def sigmoid_function(x):
    return 1/(1+np.exp(-2*x))

def tanh_function(x):
    return 2*sigmoid_function(x)-1

def derivative_tanh(x):
    return 1-((tanh_function(x))**2)

def tanh_plot():
    x = np.linspace(-10, 10, 100)
    y = tanh_function(x)
    z = derivative_tanh(x)
    plt.grid(True)
    plt.plot(x, y, color="blue", label="tanh function")
    plt.plot(x, z, color="red", label="derivative of tanh function")
    plt.show()

def main():
    tanh_plot()

if __name__=="__main__":
    main()