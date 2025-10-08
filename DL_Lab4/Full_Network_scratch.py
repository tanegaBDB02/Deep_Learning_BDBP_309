import numpy as np
import matplotlib.pyplot as plt

def dataset():
    X = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ])
    y = np.array([[0], [1], [1], [0]])
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_hat, y):
    m = y.shape[0]
    return -np.sum(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9)) / m

def init_weights(X):
    np.random.seed(42)
    W = np.random.randn(X.shape[1], 1) * 0.01
    b = 0.0
    return W, b

def train_loop(X, y, W, b, lr=0.1, epochs=1000):
    losses = []
    for epoch in range(epochs):
        z = np.dot(X, W) + b
        y_hat = sigmoid(z)
        loss = binary_cross_entropy(y_hat, y)
        losses.append(loss)
        m = X.shape[0]
        dz = (y_hat - y) / m
        dW = np.dot(X.T, dz)
        db = np.sum(dz)
        W -= lr * dW
        b -= lr * db

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss = {loss:.4f}")

    return W, b, losses

def main():
    X, y = dataset()
    W, b = init_weights(X)
    W, b, losses = train_loop(X, y, W, b)

    print("\nFinal Weights:\n", W)
    print("Final Bias:\n", b)

    y_pred = (sigmoid(np.dot(X, W) + b) > 0.5).astype(int)
    print("\nPredictions:\n", y_pred.flatten())
    print("Ground Truth:\n", y.flatten())

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title("Training Loss over 1000 iterations")
    plt.show()

if __name__ == "__main__":
    main()
