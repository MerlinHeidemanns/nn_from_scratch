from feedforward_neuralnet import *
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    N = 10000
    #K = 10
    #X = np.random.normal(-2, 2, size = (N, K))
    #b = np.random.normal(0, 0.5, size = (K, 1))
    #y = np.where(np.divide(1, 1 + np.exp(-np.matmul(X, b))) > 0.5, 1, 0)
    K = 1
    X = np.random.uniform(0, 10, size=(N, 1))
    y = np.where(np.sin(X) > 0, 1, 0 )
    #plt.scatter(X, y)
    #plt.show()
    network = Network()
    network.set_input_layer(K)
    network.add_layer(HiddenLayer(128, activation= "ReLU"))
    network.add_layer(HiddenLayer(1, activation = "sigmoid"))
    network.add_layer(CELossLayer())
    network.training(X = X, y = y, epochs = 25,minibatch = 1024, validation = 0.1, optimizer= "sgd",
                     adaptation= "decay", alpha_delta= 0.99,
                     alpha = 1)
    training_dict = network.get_training()
    plt.plot(list(range(1, 26)),training_dict["Training loss"], color='blue', label = "Training")
    plt.plot(list(range(1, 26)),training_dict["Test loss"], color='red', label = "Test")
    plt.xlabel("Epoch")
    plt.ylabel("1 - accuracy")
    plt.legend()
    plt.show()
    # #X_test = np.random.normal(-2, 2, size = (30, K))
    # #y_test = np.where(np.divide(1, 1 + np.exp(-np.matmul(X_test, b))) > 0.5, 1, 0)
    # X_test = np.random.uniform(-10, 10, size=(100, 1))
    # y_test = np.where(np.sin(X_test) > 0, 1, 0)
    # y_hat = np.transpose(np.where(network.predict(X = X_test) > 0.5, 1, 0))
    # print(np.mean(y_hat != y_test))
    #print(np.column_stack((y_test, y_hat)))
    # plt.plot(np.transpose(y), np.where(np.transpose(y_hat) > 1, 1, 0))
    # plt.show()

