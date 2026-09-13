import numpy as np
import matplotlib.pyplot as plt


class Neural_Network:

    def __init__(s, layers_dims, activations):
        s.layers_dims = layers_dims
        s.activations = activations
        s.params = {}
        s.caches = []


    def sigmoid(s, z):
        A = 1 / (1 + np.exp(-z))
        return A, z


    def relu(s, z):
        A = np.maximum(0, z)
        return A, z


    def initialize_parameters(s):
        L = len(s.layers_dims)

        for l in range(1, L):
            s.params["W" + str(l)] = (
                np.random.randn(
                    s.layers_dims[l],
                    s.layers_dims[l - 1]
                ) * 0.01
            )

            s.params["b" + str(l)] = np.zeros(
                (s.layers_dims[l], 1)
            )


    def linear_forward(s, A, W, b):
        Z = np.matmul(W, A) + b
        cache = (A, W, b)
        return Z, cache


    def linear_activation_forward(s, A, W, b, activation):

        Z, linear_cache = s.linear_forward(A, W, b)

        if activation == "relu":
            A, activation_cache = s.relu(Z)

        elif activation == "sigmoid":
            A, activation_cache = s.sigmoid(Z)

        return A, (linear_cache, activation_cache)


    def L_layer_forward(s, X):

        s.caches = []
        A = X

        for l in range(1, len(s.layers_dims)):

            A, cache = s.linear_activation_forward(
                A,
                s.params["W" + str(l)],
                s.params["b" + str(l)],
                s.activations[l]
            )

            s.caches.append(cache)

        return A, s.caches


    def compute_cost(s, AL, y):

        m = y.shape[1]

        cost = -(1 / m) * np.sum(
            y * np.log(AL + 1e-15)
            +
            (1 - y) * np.log(1 - AL + 1e-15)
        )

        return cost


    def relu_backward(s, dA, Z):

        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0

        return dZ


    def sigmoid_backward(s, dA, Z):

        A = 1 / (1 + np.exp(-Z))
        dZ = dA * A * (1 - A)

        return dZ


    def linear_backward(s, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T, dZ)

        return dA_prev, dW, db


    def linear_activation_backward(s, dA, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = s.relu_backward(dA, activation_cache)

        elif activation == "sigmoid":
            dZ = s.sigmoid_backward(dA, activation_cache)

        return s.linear_backward(dZ, linear_cache)


    def L_layer_backward(s, AL, y):

        grads = {}
        L = len(s.layers_dims) - 1

        dAL = -(y / (AL + 1e-15)) + (
            (1 - y) / (1 - AL + 1e-15)
        )

        dA_prev, dW, db = s.linear_activation_backward(
            dAL,
            s.caches[L - 1],
            s.activations[L]
        )

        grads["dA" + str(L - 1)] = dA_prev
        grads["dW" + str(L)] = dW
        grads["db" + str(L)] = db

        for l in range(L - 1, 0, -1):

            dA_prev, dW, db = s.linear_activation_backward(
                grads["dA" + str(l)],
                s.caches[l - 1],
                s.activations[l]
            )

            grads["dA" + str(l - 1)] = dA_prev
            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db

        return grads


    def update_parameters(s, grads, learning_rate):

        for l in range(1, len(s.layers_dims)):

            s.params["W" + str(l)] -= (
                learning_rate * grads["dW" + str(l)]
            )

            s.params["b" + str(l)] -= (
                learning_rate * grads["db" + str(l)]
            )


    def fit(s, X, y, iterations=1000, learning_rate=0.01):

        s.initialize_parameters()
        costs = []

        for i in range(iterations):

            AL, _ = s.L_layer_forward(X)

            cost = s.compute_cost(AL, y)

            grads = s.L_layer_backward(AL, y)

            s.update_parameters(
                grads,
                learning_rate
            )

            if i % 100 == 0:
                costs.append(cost)
                print(i, cost)

        plt.plot(
            np.arange(0, iterations, 100),
            costs
        )

        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()


    def predict(s, X):

        AL, _ = s.L_layer_forward(X)

        return (AL >= 0.5).astype(int)
    
    