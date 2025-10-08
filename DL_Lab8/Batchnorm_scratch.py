# Implement batch normalization from scratch. and layer normalization for training deep networks.
# Implement dropout to regularize neural networks from scratch.
# Implement various update rules used to optimize the neural network. You can use PyTorch for the following implementations.
# SGD, Momentum, AdaGrad, etc.

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    z_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * z_hat + beta
    cache = (x, z_hat, mean, var, gamma, beta, eps)
    return out, cache

def batchnorm_backward(dout, cache, lr=0.01):
    x, z_hat, mean, var, gamma, beta, eps = cache
    N = x.shape[0]

    # Gradients w.r.t gamma and beta
    dgamma = np.sum(dout * z_hat, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    # Update parameters
    gamma -= lr * dgamma
    beta -= lr * dbeta

    # Gradient w.r.t input
    dzhat = dout * gamma
    dvar = np.sum(dzhat * (x - mean) * -0.5 * (var + eps) ** -1.5, axis=0, keepdims=True)
    dmean = np.sum(dzhat * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)
    dx = dzhat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N
    return dx, gamma, beta

def layernorm_forward(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    z_hat = (x - mean) / np.sqrt(var + eps)
    out = gamma * z_hat + beta
    cache = (x, z_hat, mean, var, gamma, beta, eps)
    return out, cache

def layernorm_backward(dout, cache, lr=0.01):
    x, z_hat, mean, var, gamma, beta, eps = cache
    N, D = x.shape

    # Gradients w.r.t gamma and beta
    dgamma = np.sum(dout * z_hat, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    # Update parameters
    gamma -= lr * dgamma
    beta -= lr * dbeta

    # Gradient w.r.t input
    dzhat = dout * gamma
    dvar = np.sum(dzhat * (x - mean) * -0.5 * (var + eps) ** -1.5, axis=1, keepdims=True)
    dmean = np.sum(dzhat * -1 / np.sqrt(var + eps), axis=1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=1, keepdims=True)
    dx = dzhat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / D + dmean / D
    return dx, gamma, beta

def test_batchnorm():
    np.random.seed(42)
    N, D = 4, 3
    X = np.random.rand(N, D) * 5 + 10
    gamma = np.ones((1, D))
    beta = np.zeros((1, D))

    # Forward
    out, cache = batchnorm_forward(X, gamma, beta)
    print("BatchNorm Forward Output:\n", out)
    print("Mean per feature:", np.mean(out, axis=0))
    print("Variance per feature:", np.var(out, axis=0))

    # Backward
    grad_out = np.random.randn(*X.shape)
    dx, gamma, beta = batchnorm_backward(grad_out, cache)
    print("\nGradient wrt input dx:\n", dx)
    print("Updated gamma:", gamma)
    print("Updated beta:", beta)

def test_layernorm():
    np.random.seed(42)
    X = np.array([[1.0, 2.0, 3.0],
                  [2.0, 3.0, 4.0],
                  [3.0, 4.0, 5.0]])
    gamma = np.ones((1, 3))
    beta = np.zeros((1, 3))
    grad_out = np.random.randn(*X.shape)

    # Forward
    out, cache = layernorm_forward(X, gamma, beta)
    print("\nLayerNorm Forward Output:\n", out)

    # Backward
    dx, gamma, beta = layernorm_backward(grad_out, cache)
    print("\nGradient wrt input dx:\n", dx)
    print("Updated gamma:", gamma)
    print("Updated beta:", beta)

test_batchnorm()
test_layernorm()
