from nn_base import *

class BatchNorm(Module):

    def __init__(self):
        pass

    def initialize(self, optimization, initialization, regularization):
        self.next.initialize(optimization, initialization, regularization)
        self.beta = n
        if self.initialization is None:
            self.beta = np.random.normal(loc=0, scale=0.2, size=(self.k, 1))
            self.gamma = np.random.normal(loc=0, scale=0.2, size=(self.k, 1))
        elif self.initialization == "normalized":
            self.beta = np.random.uniform(low=-np.sqrt(2 / (self.k)), high=np.sqrt(2 / (self.k)),
                                       size=(self.k, 1))
            self.gamma = np.random.uniform(low=-np.sqrt(2 / (self.k)), high=np.sqrt(2 / (self.k)),
                                       size=(self.k, 1))

    def __call__(self, other, d = None):
        if isinstance(other, Module):
            if self.next is None:
                self.next = other
                other.k = self.k
            else:
                self.next(other)

    def forward(self, y, parameters):
        self.mu = np.mean(self.input, axis = 1)
        self.sigma2 = np.mean(np.square(np.transpose(self.input) - self.mu), axis = 0)
        self.input_transformed = np.transpose(np.divide(np.transpose(self.input) - self.mu, np.sqrt(self.sigma2 + 0.0001) ))
        self.next.input = self.beta + np.multiply(self.gamma, self.input_transformed)
        self.next.forward(y=y, parameters = parameters)

    def backward(self):
        self.next.backward()
        self.delta_delta = np.sum(np.multiply(self.next.gradient, self.input_transformed),  axis = 0, keepdim = True)
        self.delta_beta  = np.sum(self.next.gradient, axis = 0, keepdim = True)
        self.delta_sigma = np.sum(np.multiply(self.next.gradient, self.input - self.mu)



    def predict(self):
        self.next.input = self.input

    def update(self, parameters = None):
        self.beta = self.beta - 1/self.input * np.sum(self.next.gradient, axis = 0, keepdim = True)

