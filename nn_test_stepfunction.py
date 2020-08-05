from feedforward_neuralnet import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    K = 1
    N = 300
    X = np.random.uniform(-10, 10, size = (N, K))
    b = np.ones(shape=(1,1))
    y = np.sign(np.matmul(X, b) + 3)
    #plt.scatter(X, y)
    #plt.show()
    network = Network()
    network.set_input_layer(K)
    network.add_layer(HiddenLayerSigmoid(256, 0.01))
    network.add_layer(HiddenLayerSigmoid(32, 0.01))
    network.add_layer(HiddenLayerSigmoid(1, 0.01))
    network.add_layer(MSELossLayer())
    network.training(X = X, y = y, epochs = 100,minibatch = 25, validation = 0.1)
    x_test = np.arange(-10, 10, 0.1).reshape(200, 1)
    y_hat = network.predict(X = x_test)
    plt.plot(x_test, np.where(np.transpose(y_hat) > 0.5, 1, 0))
    plt.show()