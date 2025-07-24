# Consider the following two networks. W is a matrix, x is a vector, z is a vector, and a is a vector.
# y^ is a scalar and a final prediction. Initialize x, w randomly, z is a dot product of x and w, a is ReLU(z).
# Initialize X and W randomly. Every neuron has a bias term.

# Implement forward pass for the above two networks. Print activation values for each neuron at each layer.
# Print the loss value (y^).
# Implement the forward pass using vectorized operations, i.e. W should be a matrix, x, z and a are vectors.
# The implementation should not contain any loops.


#hardcoded version of first network
import numpy as np

# def initialization():
#     weights=np.array([0.001,0.01,-0.005,-1.2])
#     X = np.array([0.2, -1.3, 2, 0.01])
#     prod=weights*X
#     z=sum(prod)
#     return z
#
# def softmax(z):
#     a=np.exp(z)/(np.sum(np.exp(z)))
#     return a
#
# def main():
#     z=initialization()
#     a = softmax(z)
#     print("Z value:\n", z)
#     print("Softmax:\n", a)
#
# if __name__ == "_main_":
#     main()
#
#
#
# # #user defined version of first network
# def initialization():
#     rows,columns=4,1
#     W=np.random.rand(rows,columns)
#     weights=np.round(W,2)
#     return weights
#
# def computation(weights):
#     X=[]
#     print("How many numbers do you want?")
#     n=int(input())
#     for i in range(n):
#         print("Enter element",i+1,":",end=' ')
#         elem=float(input())
#         X.append(elem)
#     z=np.dot(X,weights)
#     print("X vector:\n", X)
#     return z
#
# def softmax(z):
#     a=np.exp(z)/(sum(np.exp(z)))
#     return a
#
# def main():
#     weights = initialization()
#     z = computation(weights)
#     a = softmax(z)
#     print("Weights:\n", weights)
#     print("Z value:\n", z)
#     print("Softmax:\n", a)
#
# if __name__ == "_main_":
#     main()



#user defined version of second network
from DL_Lab1.ReLu import Relu_function
from math import exp

def user_input():
    x=[]
    print("How many inputs do you want?")
    n = int(input())
    for i in range(n):
        elem = float(input(f"Enter element {i+1}:"))
        x.append(elem)
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

def forward_pass(x,layer_size):
    input_data=x
    idx=0
    for neurons in layer_size:
        W=generate_weights(input_data.shape[1],neurons)
        z=np.dot(input_data,W)
        a=Relu_function(z)
        print(f"After Layer {idx+1} ReLU Activation:\n",a)
        input_data=a
    return input_data

def output_layer(x, output_neurons):
    W=generate_weights(x.shape[1],output_neurons)
    z=np.dot(x,W)
    z_flat=z.flatten()
    exponents = [exp(i) for i in z_flat]
    sum_of_exponents = sum(exponents)
    S=[round(i / sum_of_exponents, 4) for i in exponents]
    print("Softmax Output:\n",S)

def main():
    print()
    print("Performing the Forward Pass on the second network")
    x,hidden_layers,neurons,output_neurons =user_input()
    hidden_output=forward_pass(x,neurons)
    output_layer(hidden_output,output_neurons)

if __name__ == "__main__":
    main()