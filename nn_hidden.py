from nn_base import *
import numpy as np

class HiddenLayer(Module):

    def __init__(self, d):
        super().__init__()
        self.d = d

    def __call__(self, other):
        if isinstance(other, Module):
            if self.next is None:
                self.next = other
                other.k = self.d
            else:
                self.next(other)

    def show_architecture(self):
        print(self.__class__.__name__, self.k, self.d)
        self.next.show_architecture()

    def initialize(self):
        self.next.initialize()
        self.b = np.random.normal(loc=0, scale=0.2, size=(self.d, 1))
        self.w = np.random.normal(loc = 0, scale = 0.2, size = (self.d, self.k))

    def _compute_a(self):
        self.a = np.add(self.b, np.matmul(self.w, self.input))

    def _compute_h(self):
        pass

    def forward(self, y):
        self._compute_a()
        self._compute_h()
        return(self.next.forward(y))

    def backward(self):
        self.next.backward()

    def get_accuracy(self, y):
        self.forward(y)
        return(self.next.get_accuracy(y))

    def update(self, parameters = None):
        self.next.update(parameters)
        self.n = self.input.shape[1]
        self.b_update = - 1/self.n * np.sum(self.delta, axis = 1, keepdims = True)
        self.w_update = - 1/self.n * np.matmul(self.delta, np.transpose(self.input))
        self.b = np.add(self.b, parameters["alpha"] * self.b_update)
        self.w = np.add(self.w, parameters["alpha"] * self.w_update)

class ReLUHiddenLayer(HiddenLayer):

    def __init__(self, d):
        super().__init__(d = d)

    def _relu(self, x):
        return max([0, x])

    def _grelu(self, x):
        return np.where(x > 0, 1, 0)

    def _compute_h(self):
        _vrelu = np.vectorize(self._relu)
        self.next.input = _vrelu(self.a)

    def backward(self):
        super().backward()
        self.delta    = np.multiply(self._grelu(self.a), self.next.gradient)
        self.gradient = np.matmul(np.transpose(self.w), self.delta)