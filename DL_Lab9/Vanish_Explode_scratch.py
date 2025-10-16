# Write a program to simulate vanishing & exploding gradient problems.
# Plot the gradient values to demonstrate the issues with training a very deep network.
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def sigmoid_derivative(x):
#     s = sigmoid(x)
#     return s * (1 - s)
#
# def simulate_gradients(layers=50, weight_scale=0.1):
#     np.random.seed(42)
#     W_list = [np.random.randn(1, 1) * weight_scale for _ in range(layers)]
#
#     x = np.array([[1.0]])
#     y = np.array([[0.0]])  # target
#     grad_magnitudes = []
#
#     activations = [x]
#     z_values = []
#     for W in W_list:
#         z = activations[-1] @ W
#         z_values.append(z)
#         a = sigmoid(z)
#         activations.append(a)
#
#     dL_da = 2 * (activations[-1] - y)  # derivative of MSE
#     da = dL_da
#     for i in reversed(range(layers)):
#         dz = da * sigmoid_derivative(z_values[i])
#         dW = activations[i].T @ dz
#         grad_magnitudes.append(abs(dW[0, 0]))  # track magnitude
#         da = dz @ W_list[i].T
#
#     # Reverse so first layer gradient is first
#     grad_magnitudes = grad_magnitudes[::-1]
#     return grad_magnitudes
#
# def main():
#     layers = 50
#     grad_vanishing = simulate_gradients(layers, weight_scale=0.01)
#     grad_stable = simulate_gradients(layers, weight_scale=0.1)
#     grad_exploding = simulate_gradients(layers, weight_scale=1.5)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(grad_vanishing, label='Vanishing Gradients (small weights)')
#     plt.plot(grad_stable, label='Stable Gradients (medium weights)')
#     plt.plot(grad_exploding, label='Exploding Gradients (large weights)')
#     plt.xlabel("Layer")
#     plt.ylabel("Gradient magnitude of each layer")
#     plt.title("Vanishing & Exploding Gradients in Deep Network")
#     plt.yscale('log')  # log scale for clarity
#     plt.legend()
#     plt.show()
#
# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt

def dropout_forward(X, p=0.5, training=True):
    if training:
        mask = (np.random.rand(*X.shape) > p).astype(np.float32)
        out = (X * mask) / (1 - p)
    else:
        mask = None
        out = X
    return out, mask

def dropout_backward(grad_output, mask, p=0.5):
    dx = (grad_output * mask) / (1 - p)
    return dx

def main():
    np.random.seed(0)

    X = np.array([[1.0, 2.0, 3.0],
                  [2.5, 7.7, 6.4],
                  [5.8, 6.0, 9.2]])

    p = 0.5  # dropout probability
    training = True

    # Forward pass (training)
    out_train, mask = dropout_forward(X, p=p, training=training)
    print("Train forward output:\n", out_train)

    # Backward pass
    grad_out = np.ones_like(X)
    dx = dropout_backward(grad_out, mask, p=p)
    print("\nBackward gradient:\n", dx)

    # Forward pass
    out_test, _ = dropout_forward(X, p=p, training=False)
    print("\nTest forward output:\n", out_test)

    plt.figure(figsize=(6, 4))
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Mask (0=dropped, 1=kept)')
    plt.title(f'Dropout Mask Visualization (p={p})')
    plt.xlabel("Neuron index")
    plt.ylabel("Sample index")
    plt.show()

if __name__ == "__main__":
    main()
