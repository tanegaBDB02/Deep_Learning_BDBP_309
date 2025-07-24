import matplotlib.pyplot as plt
import numpy as np

def leaky_relu(x,alpha):
    y=np.abs(x)
    return (((1+alpha)*x)+((1-alpha)*y))/2

def derivative_leaky_relu(x, alpha):
    result = []
    for i in x:
        if i>0:
            result.append(1)
        else:
            result.append(alpha)
    return result

def leaky_relu_plot():
    x = np.linspace(-10, 10, 100)
    y = leaky_relu(x, alpha=0.1)
    z=derivative_leaky_relu(x,alpha=0.01)
    plt.grid(True)
    plt.plot(x, y, color="blue", label="Leaky Relu function")
    plt.plot(x, z, color="red", label="derivative of Leaky Relu function")
    plt.show()

def main():
    leaky_relu_plot()

if __name__ == '__main__':
    main()