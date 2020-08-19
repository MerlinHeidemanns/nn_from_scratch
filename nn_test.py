from nn_base import *
from nn_hidden import *
from nn_loss import *
from nn_network import *

if __name__ == '__main__':

    N = np.random.randint(2000, 5000)
    K = np.random.randint(2, 10)
    X = np.random.normal(loc = -0.5, scale = 0.5, size = (N, K))
    b = np.random.uniform(-2, 1, size = (K, 1))
    y_star = np.divide(np.ones(shape=(N, 1)),np.add(np.ones(shape=(N, 1)), np.exp(-np.matmul(X, b))))
    y = np.where(y_star > 0.5, 1, 0)
    nn = Network()
    nn(ReLUHiddenLayer(128), K)
    nn(ReLUHiddenLayer(1))
    nn(BCELossLayer())
    nn.show_architecture()
    nn.initialize()
    nn.train(X, y, epochs = 20, minibatch = 1024, validation = 0.1, parameters={"alpha": 0.1})
    nn.show_error()
    #print(y)