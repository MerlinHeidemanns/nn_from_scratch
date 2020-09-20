from nn_base import *
import numpy as np

class LossLayer(Module):

    def __init__(self):
        super().__init__()

    def show_architecture(self):
        print(self.__class__.__name__)

    def forward(self, y = None, parameters = None):
        self.y = np.transpose(y)

    def backward(self):
        pass

    def get_accuracy(self, y):
        return(self.forward(y))

    def predict(self):
        return(self.input)

    def initialize(self, optimization, initialization, regularization):
        pass

    def update(self):
        pass

class MSELossLayer(LossLayer):

    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self, y, parameters = None):
        super().forward(y=y)
        self.loss = np.mean(np.square(self.input - self.y))
        return(self.loss)

    def backward(self):
        self.gradient = np.subtract(self.input, self.y)

class BCELossLayer(MSELossLayer):

    def __init__(self):
        super().__init__()

    def forward(self, y, parameters = None):
        super().forward(y=y)
        self.loss = np.mean(np.where(np.where(self.input > 0.5, 1, 0) != self.y, 1, 0))
        return(self.loss)
