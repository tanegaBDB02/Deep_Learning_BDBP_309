from DL_Lab1.ReLu import Relu_function, derivative_Relu
from math import exp
import numpy as np

def user_input():
    x=[]
    print("How many input features do you want?")
    n = int(input())
    for i in range(n):
        inp=float(input(f"Enter input value {i+1}:"))
        x.append(inp)
    x=np.array(x).reshape(1,-1)
    print("X vector:\n", x)

    print("How many hidden layers do you want?")
    hidden_layers=int(input())
    print("Hidden layers:", hidden_layers)

    neurons=[]
    for j in range(hidden_layers):
        print(f"How many neurons do you want in layer {j+1}?")
        n=int(input())
        neurons.append(n)

    print("How many output neurons?")
    output_neurons = int(input())

    return x,hidden_layers,neurons,output_neurons

def generate_weights(input_size,output_size):
    W=(np.random.rand(input_size,output_size)-0.5)*0.2
    #weights = np.round(W,2)
    print(f"Generated Weights [{input_size}x{output_size}]:\n", W)
    #print("Weights:\n", weights)

    return W

def bias_input(layer,length):
    bias=[]
    for i in range(length):
        print(f"Enter the bias for neuron [{i+1}] in {layer}")
        b=float(int(input()))
        bias.append(b)

    return np.array(bias).reshape(1,-1)

def forward_pass(x,layer_size):
    input_data=x
    index=0
    zs = []  # storing z values for all layers
    activations = [x]  # storing activations including input
    weights = []
    biases = []

    for neurons in layer_size:
        W=generate_weights(input_data.shape[1],neurons)
        b = bias_input(f"Hidden Layer {index+1}",neurons)
        weights.append(W)
        biases.append(b)
        z=np.dot(input_data,W)+b
        a=Relu_function(z)
        zs.append(z)
        activations.append(a)
        input_data=a
    return activations,zs,weights,biases

def output_layer(x, output_neurons):
    W = generate_weights(x.shape[1], output_neurons)
    b = np.zeros((1, output_neurons))   # initialize bias
    z = np.dot(x, W) + b
    exp_z = np.exp(z - np.max(z))       # stable softmax
    S = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    print("Softmax Output (y_hat):\n", S)
    return S, W, b, z

def get_target(output_neurons): #creates one-hot target
    true_class = int(input(f"Enter the correct class index (0 to {output_neurons - 1}): "))
    y = np.zeros((1, output_neurons))
    y[0, true_class] = 1
    return y

def derivative_Relu(z):
    return (z > 0).astype(float)

def backpropagation(activations,weights,biases,zs,y_hat,y,lr=0.01):
    """ activations: list of activations for each layer (including input)
        weights: list of weight matrices
        biases: list of bias vectors
        zs: list of pre-activation values (z = XW + b)
        y_hat: predicted output (softmax probabilities)
        y: true labels (one-hot vector)
        lr: learning rate
    """
    L=len(weights)
    delta=y_hat-y

    for l in reversed(range(L)):
        a_previous=activations[l]
        dW=np.dot(a_previous.T,delta)
        db = np.sum(delta, axis=0, keepdims=True)  # bias gradient

        print(f"\nLayer {l + 1} gradients:")
        print("dW:\n", dW)
        print("db:\n", db)

        weights[l]-=lr*dW
        biases[l]-=lr*db
        if l>0:
            delta = (delta @ weights[l].T) * (zs[l - 1] > 0).astype(float)
    print("\nUpdated Weights of hidden/output layers:", [np.round(w, 6) for w in weights])
    print("Updated Bias of hidden/output layers:", [np.round(b, 6) for b in biases])
    return weights, biases

def main():
    x, hidden_layers, neurons, output_neurons = user_input()
    activations, zs, weights, biases, = forward_pass(x, neurons)

    W_out = generate_weights(activations[-1].shape[1], output_neurons)
    b_out = np.zeros((1, output_neurons))
    z_out = np.dot(activations[-1], W_out) + b_out

    exp_z = np.exp(z_out - np.max(z_out))
    y_hat = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    print("Softmax Output (y_hat):\n", y_hat)

    y = get_target(output_neurons)
    print("Target y:\n", y)

    weights.append(W_out)
    biases.append(b_out)
    zs.append(z_out)
    activations.append(y_hat)  # final output activation

    weights, biases = backpropagation(activations, weights, biases, zs, y_hat, y, lr=0.01)


if __name__ == "__main__":
    main()


