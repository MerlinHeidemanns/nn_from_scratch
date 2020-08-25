
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
from nn_base import *
from nn_hidden import *
from nn_loss import *
from nn_network import *

if __name__ == '__main__':

    # "Simple" way (will work as long as you never shuffle X and Y)
    data = sio.loadmat("data/hw3_data-3.mat")
    X1, Y1 = data['X1'], data['Y1']
    Y1     = Y1 / 256
    X1_adj     = X1 / np.max(X1)
    nn = Network()
    nn(ReLUHiddenLayer(256), 2)
    nn(ReLUHiddenLayer(128))
    nn(ReLUHiddenLayer(64))
    nn(SigmoidHiddenLayer(1))
    nn(MSELossLayer())
    nn.train(X1_adj, Y1, epochs=200, minibatch=1024, validation=0.1,
             parameters={"epsilon": 0.7,
                         "epsilon_adjustment": 0.95,
                         "alpha_momentum": 0.5,
                         "rho": 0.1},
             optimization="nesterov_momentum", initialization="normalized")
    nn.show_error()
    y_hat = nn.predict(X1_adj)
    y_hat = y_hat * 256
    img1 = np.zeros((100, 76))
    for i in range(X1.shape[0]):
        xpixel, ypixel = int(X1[i][0]), int(X1[i][1])
        img1[xpixel, ypixel] = y_hat[i]
    plt.imshow(img1, cmap="gray")
    plt.show()
