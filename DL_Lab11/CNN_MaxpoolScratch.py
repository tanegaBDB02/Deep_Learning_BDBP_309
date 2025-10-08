# Implement convolution operations from scratch. Assume a 3x3 kernel and apply it on an input image of 32x32.
# Implement maxpool operation from scratch.

import numpy as np

def conv(kernel,input,stride=1):
    input = np.array(input)
    kernel = np.array(kernel)
    h,w=input.shape
    kh,kw=kernel.shape
    output_h=(h-kh)//stride + 1
    output_w=(w-kw) //stride+ 1

    output=np.zeros((output_h,output_w))
    for i in range(output_h):
        for j in range(output_w):
            block=input[i*stride:i*stride+kh,j*stride:j*stride+kw]
            output[i,j]=np.sum(block*kernel)

    print ("\nConvolution output:\n",output)
    return output

def maxpool(input,pool_size=2,stride=1):
    input = np.array(input)
    h,w=input.shape
    ph,pw=pool_size,pool_size
    pool_output_h=(h-ph)//stride+1
    pool_output_w=(w-pw) //stride+1

    pool_output=np.zeros((pool_output_h,pool_output_w))
    for i in range(pool_output_h):
        for j in range(pool_output_w):
            block=input[i*stride:i*stride+ph, j*stride:j*stride+pw]
            pool_output[i,j]=np.max(block)

    print ("\nMaxpool output:\n",pool_output)
    return pool_output


def main():
    # input_data=[[3,0,1,2,7,4],
    #         [1,5,8,9,3,1],
    #         [2,7,2,5,1,3],
    #         [0,1,3,1,7,8],
    #         [4,2,1,6,2,8],
    #         [2,4,5,2,3,9]]
    #
    # kernel=[[1,0,-1],
    #          [1,0,-1],
    #          [1,0,-1]]

    input_data = np.random.randint(0, 256, (32, 32))
    kernel = np.random.uniform(-1, 1, (3, 3))
    print ("Randomized Input image:\n", input_data)
    print("Random kernel:\n", kernel)

    #Convolution
    conv_out = conv(kernel,input_data,stride=1)
    print("Convolution output shape:", conv_out.shape)

    #MaxPooling
    pooled_out = maxpool(conv_out, pool_size=2, stride=2)
    print("MaxPool output shape:", pooled_out.shape)


if __name__ == "__main__":
    main()


