from nn_base import *
import numpy as np

class ActivationLayer(Module):

    def __init__(self):
        super().__init__()


    def __call__(self, other, d = None):
        if isinstance(other, Module):
            if self.next is None:
                self.next = other
                other.k = self.k
            else:
                self.next(other)

    def backward(self):
        self.next.backward()



class ReLUActivationLayer(ActivationLayer):

    def __init__(self):
        super().__init__()

    def _relu(self, x):
        return max([0, x])

    def _grelu(self, x):
        return np.where(x > 0, 1, 0)

    def predict(self):
        _vrelu = np.vectorize(self._relu)
        self.next.input = _vrelu(self.input)
        return(self.next.predict())

    def get_accuracy(self, y):
        _vrelu = np.vectorize(self._relu)
        self.next.input = _vrelu(self.input)
        return (self.next.get_accuracy(y))

    def forward(self, y = None, parameters = None):
        _vrelu = np.vectorize(self._relu)
        self.next.input = _vrelu(self.input)
        self.next.forward(y = y, parameters = parameters)

    def backward(self):
        super().backward()
        self.gradient = np.multiply(self._grelu(self.input), self.next.gradient)

class SigmoidActivationLayer(ActivationLayer):

    def __init__(self):
        super().__init__()

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _gsigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def forward(self, y = None, parameters = None):
        _vsigmoid = np.vectorize(self._sigmoid)
        self.next.input = _vsigmoid(self.input)
        self.next.forward(y = y, parameters = parameters)

    def get_accuracy(self, y):
        _vsigmoid = np.vectorize(self._sigmoid)
        self.next.input = _vsigmoid(self.input)
        return (self.next.get_accuracy(y))

    def backward(self):
        super().backward()
        _vgsigmoid = np.vectorize(self._gsigmoid)
        self.gradient = np.multiply(_vgsigmoid(self.input), self.next.gradient)


    def predict(self):
        _vsigmoid = np.vectorize(self._sigmoid)
        self.next.input = _vsigmoid(self.input)
        return(self.next.predict())