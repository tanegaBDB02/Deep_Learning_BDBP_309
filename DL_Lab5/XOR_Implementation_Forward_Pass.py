#Implement a 3-layer (input layer, hidden layer and output layer) neural network from scratch for the XOR operation.
# This includes implementing forward and backward passes from scratch.

from DL_Lab1.ReLu import Relu_function
import numpy as np

def user_input(n):
    x1=[]
    x2=[]
    print("Enter inputs for x1:")
    for i in range(n):
        val = float(input(f"x1[{i+1}]: "))
        x1.append(val)

    print("Enter inputs for x2:")
    for i in range(n):
        val = float(input(f"x2[{i+1}]: "))
        x2.append(val)

    X = np.column_stack((x1,x2))
    print("X vector:\n", X)
    return X

def weight_input(layer,shape):
    weights=[]
    for i in range(shape[0]):
        row=[]
        for j in range(shape[1]):
            print(f"Enter the weight for weights[{i+1}][{j+1}] of the {layer} layer")
            w=float(int(input()))
            row.append(w)
        weights.append(row)
    return np.array(weights)

def bias_input(layer,length):
    bias=[]
    for i in range(length):
        print(f"Enter the bias for [{i+1}] of the {layer} layer")
        b=float(int(input()))
        bias.append(b)

    return np.array(bias)

def forward_pass(X,hidden_weights,hidden_bias,outer_weights,outer_bias):
    z_hidden=np.dot(X,hidden_weights.T)+hidden_bias
    a_hidden=Relu_function(z_hidden)

    z_outer=np.dot(a_hidden,outer_weights)+outer_bias
    a_outer=Relu_function(z_outer)

    print(f"Hidden layer ReLU Activation:\n",a_hidden)
    print(f"Outer layer ReLU Activation:\n", a_outer)

def main():
    print("Performing the XOR Implementation with Forward pass")
    X=user_input(n=4)
    print("\nEnter weights for Hidden Layer (2 neurons, 2 inputs):")
    hidden_weights=weight_input("Hidden",(2,2))
    hidden_bias=bias_input("Hidden",2)

    print ("\nEnter weights for Output Layer (1 neuron, 2 hidden outputs):")
    outer_weights=weight_input("Outer",(2,1))
    outer_bias=bias_input("Outer",1).reshape(1,)

    forward_pass(X,hidden_weights,hidden_bias,outer_weights,outer_bias)

if __name__ == "__main__":
    main()