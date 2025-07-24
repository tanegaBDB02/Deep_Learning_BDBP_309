import matplotlib.pyplot as plt
import numpy as np


def Relu_function(x):
    return (x + np.abs(x)) / 2


def derivative_Relu(x):
    result = []
    for i in x:
        if i <= 0:
            result.append(0)
        else:
            result.append(1)
    return result


def relu_plot():
    x = np.linspace(-10, 10, 100)
    y = Relu_function(x)
    z = derivative_Relu(x)
    plt.grid(True)
    plt.plot(x, y, color="blue", label="relu function")
    plt.plot(x, z, color="red", label="derivative of relu function")
    plt.show()


def main():
    relu_plot()


if __name__ == '__main__':
    main()
