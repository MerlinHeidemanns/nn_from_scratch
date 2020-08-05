from feedforward_neuralnet import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    K = 3
    N = 1000
    X = np.random.normal(-10, 10, size = (N, K))
    b = np.random.normal(0, 0.3, size = (K, 1))
    y = np.where(np.divide(1, 1 + np.exp(-np.matmul(X, b))) > 0.5, 1, 0)
    network = Network()
    network.set_input_layer(K)
    network.add_layer(HiddenLayerSigmoid(512))
    network.add_layer(HiddenLayerSigmoid(128))
    network.add_layer(HiddenLayerSigmoid(1))
    network.add_layer(MSELossLayer())
    network.set_optimizer("sgd")
    network.training(X = X, y = y, epochs = 20,minibatch = 512, validation = 0.1, optimizer= "sgd", alpha = 1)
    X_test = np.random.normal(-10, 10, size = (300, K))
    y_test = np.where(np.divide(1, 1 + np.exp(-np.matmul(X_test, b))) > 0.5, 1, 0)
    y_hat = np.transpose(np.where(network.predict(X = X_test) > 0.5, 1, 0))
    print(np.mean(y_hat != y_test))
    # plt.plot(np.transpose(y), np.where(np.transpose(y_hat) > 1, 1, 0))
    # plt.show()
