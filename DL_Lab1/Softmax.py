from math import exp

def softmax(vector):
    exponents=[exp(i) for i in vector]
    sum_of_exponents=sum(exponents)
    S=[round(i/sum_of_exponents,4) for i in exponents]
    return S

def derivative_softmax(S):
    Jacobian=[]
    n=len(S)
    for i in range(n):
        row=[]
        for j in range(n):
            if i==j:
                val=S[i]*(1-S[i])
            else:
                val=-S[i]*S[j]
            row.append(round(val,4))
        Jacobian.append(row)
    return Jacobian


def main():
    vec = [2.0, 1.0, 0.1]
    S = softmax(vec)
    print("Softmax:",S)

    J = derivative_softmax(S)
    print("Jacobian of Softmax:")
    print(J)

if __name__ == '__main__':
    main()