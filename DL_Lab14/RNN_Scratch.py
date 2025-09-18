import numpy as np

#activation function
def sigmoid_function(x):
    return 1/(1+np.exp(-2*x))

def tanh_function(x):
    return 2*sigmoid_function(x)-1

def ht_calculation(Whh,h_previous,Wxh,input_previous):
    forward_calc=np.dot(Whh,h_previous)+np.dot(Wxh,input_previous)
    ht=tanh_function(forward_calc)
    return ht

def output_calculation(Why, ht):
    yt=np.dot(Why,ht)
    return yt

def matrix(rows,columns,name):
    print(f"Enter values for {name} ({rows}x{columns}) row by row, separated by spaces:")
    mat=[]
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        mat.append(row)
    return np.array(mat)

def main():
    n=int(input("How many inputs do you want?"))
    input_dim=int(input("Enter input dimension:"))
    hidden_dim=int(input("Enter hidden dimension:"))
    output_dim=int(input("Enter output dimension:"))
    T=int(input("Enter number of time steps:"))

    Wxh=matrix(hidden_dim,input_dim,"Wxh")
    Whh=matrix(hidden_dim,hidden_dim,"Whh")
    Why=matrix(output_dim,hidden_dim,"Why")

    print(f"Enter initial hidden state h0 (size {hidden_dim}):")
    h_previous = np.array(list(map(float,input().split()))).reshape(hidden_dim, 1)

    inputs = []
    for t in range(T):
        print(f"Enter input vector at time {t} (size {input_dim}):")
        x_t = np.array(list(map(float, input().split()))).reshape(input_dim, 1)
        inputs.append(x_t)

    for t in range(T):
        h_previous=ht_calculation(Whh, h_previous,Wxh,inputs[t])
        yt=output_calculation(Why, h_previous)
        print(f"\nTime step {t}:")
        print("Hidden state h_t:\n",h_previous)
        print("Output y_t:\n",yt)

if __name__=="__main__":
    main()

