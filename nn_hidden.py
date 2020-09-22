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

    def initialize(self, optimization, initialization, regularization):
        self.next.initialize(optimization, initialization, regularization)
        self.initialization = initialization
        self.regularization = regularization
        self.b = np.zeros(shape=(self.d, 1))
        if self.initialization is None:
            self.w = np.random.normal(loc=0, scale=0.2, size=(self.d, self.k))
        elif self.initialization == "normalized":
            self.w = np.random.uniform(low=-np.sqrt(6 / (self.d + self.k)), high=np.sqrt(6 / (self.d + self.k)),
                                       size=(self.d, self.k))
        self.optimization = optimization
        if self.optimization == "momentum" or self.optimization == "nesterov_momentum":
            self.v_b = np.zeros(shape=(self.d, 1))
            self.v_w = np.zeros(shape=(self.d, self.k))
        elif self.optimization == "adagrad" or self.optimization == "rmsprop":
            self.r_b = np.zeros(shape=(self.d, 1))
            self.r_w = np.zeros(shape=(self.d, self.k))
        elif self.optimization == "rmsprop_nesterov":
            self.v_b = np.zeros(shape=(self.d, 1))
            self.v_w = np.zeros(shape=(self.d, self.k))
            self.r_b = np.zeros(shape=(self.d, 1))
            self.r_w = np.zeros(shape=(self.d, self.k))
        elif self.optimization == "adam":
            self.s_b = np.zeros(shape=(self.d, 1))
            self.s_w = np.zeros(shape=(self.d, self.k))
            self.r_b = np.zeros(shape=(self.d, 1))
            self.r_w = np.zeros(shape=(self.d, self.k))
            self.t = 0
        if self.regularization == "batch_norm":
            self.beta = np.random.uniform(low = -np.sqrt(2/self.d), high = np.sqrt(2/self.d), size = (self.d, 1))
            self.gamma = np.random.uniform(low = -np.sqrt(2/self.d), high = np.sqrt(2/self.d), size = (self.d, 1))


    def _compute_output(self):
        if self.optimization == "nesterov_momentum" or self.optimization == "rmsprop_nesterov":
            alpha_momentum = self.parameters["alpha_momentum"]
            self.output = np.add(np.add(self.b, alpha_momentum * self.v_b),
                            np.matmul(np.add(self.w, alpha_momentum * self.v_w), self.input))
        else:
            self.output = np.add(self.b, np.matmul(self.w, self.input))

    def forward(self, y=None, parameters=None):
        self.parameters = parameters
        self._compute_output()
        self.next.input = self.output
        self.next.forward(y=y, parameters=self.parameters)

    def backward(self):
        self.next.backward()
        self.gradient = np.matmul(np.transpose(self.w), self.next.gradient)

    def predict(self):
        self.next.input = np.add(self.b, np.matmul(self.w, self.input))
        return (self.next.predict())

    def get_accuracy(self, y):
        self.next.input = np.add(self.b, np.matmul(self.w, self.input))
        return (self.next.get_accuracy(y))

    def compute_update(self):
        self.n = self.input.shape[1]
        self.b_update = 1 / self.n * np.sum(self.next.gradient, axis=1, keepdims=True)
        self.w_update = 1 / self.n * np.matmul(self.next.gradient, np.transpose(self.input))

    def update(self):
        self.next.update()
        self.compute_update()
        if self.optimization == "sgd":
            self.b = np.add(self.b, - self.parameters["epsilon"] * self.b_update)
            self.w = np.add(self.w, - self.parameters["epsilon"] * self.w_update)
        elif self.optimization == "momentum":
            self.update_momentum()
        elif self.optimization == "nesterov_momentum":
            self.update_nesterov_momentum()
        elif self.optimization == "adagrad":
            self.update_adagrad()
        elif self.optimization == "rmsprop":
            self.update_rmsprop()
        elif self.optimization == "rmsprop_nesterov":
            self.update_rmsprop_nesterov()
        elif self.optimization == "adam":
            self.update_adam()

    def update_momentum(self):
        """
        Common alpha_momentum values are 0.5, 0.9, 0.99
        :param parameters: Dictionary of parameters
        :return: None
        """
        alpha_momentum = self.parameters["alpha_momentum"]
        epsilon = self.parameters["epsilon"]
        self.v_b = np.subtract(alpha_momentum * self.v_b, epsilon * self.b_update)
        self.v_w = np.subtract(alpha_momentum * self.v_w, epsilon * self.w_update)
        self.b = np.add(self.b, self.v_b)
        self.w = np.add(self.w, self.v_w)

    def update_nesterov_momentum(self):
        alpha_momentum = self.parameters["alpha_momentum"]
        epsilon = self.parameters["epsilon"]
        self.v_b = np.subtract(alpha_momentum * self.v_b, epsilon * self.b_update)
        self.v_w = np.subtract(alpha_momentum * self.v_w, epsilon * self.w_update)
        self.b = np.add(self.b, self.v_b)
        self.w = np.add(self.w, self.v_w)

    def update_adagrad(self):
        delta = pow(10, -7)
        epsilon = self.parameters["epsilon"]
        self.r_b = np.add(self.r_b, np.square(self.b_update))
        self.r_w = np.add(self.r_w, np.square(self.w_update))
        b_update = - np.multiply(np.divide(epsilon, np.add(delta, np.sqrt(self.r_b))), self.b_update)
        w_update = - np.multiply(np.divide(epsilon, np.add(delta, np.sqrt(self.r_w))), self.w_update)
        self.b = np.add(self.b, b_update)
        self.w = np.add(self.w, w_update)

    def update_rmsprop(self):
        """
        delta   = stabilizing constant
        rho     = decay rate of past gradients
        epsilon = learning rate
        """
        delta = pow(10, -6)
        rho = self.parameters["rho"]
        epsilon = self.parameters["epsilon"]
        self.r_b = np.add(rho * self.r_b, (1 - rho) * np.square(self.b_update))
        self.r_w = np.add(rho * self.r_w, (1 - rho) * np.square(self.w_update))
        b_update = - np.multiply(np.divide(epsilon, np.add(delta, np.sqrt(self.r_b))), self.b_update)
        w_update = - np.multiply(np.divide(epsilon, np.add(delta, np.sqrt(self.r_w))), self.w_update)
        self.b = np.add(self.b, b_update)
        self.w = np.add(self.w, w_update)

    def update_rmsprop_nesterov(self):
        delta = pow(10, -6)
        alpha_momentum = self.parameters["alpha_momentum"]
        rho = self.parameters["rho"]
        epsilon = self.parameters["epsilon"]
        self.r_b = np.add(rho * self.r_b, (1 - rho) * np.square(self.b_update))
        self.r_w = np.add(rho * self.r_w, (1 - rho) * np.square(self.w_update))
        # velocity update
        self.v_b = np.subtract(alpha_momentum * self.v_b,
                               np.multiply(np.divide(epsilon, np.add(delta, np.sqrt(self.r_b))), self.b_update))
        self.v_w = np.subtract(alpha_momentum * self.v_w,
                               np.multiply(np.divide(epsilon, np.add(delta, np.sqrt(self.r_w))), self.w_update))
        # update
        self.b = np.add(self.b, self.v_b)
        self.w = np.add(self.w, self.v_w)

    def update_adam(self):
        rho1 = self.parameters["rho1"]
        rho2 = self.parameters["rho2"]
        epsilon = self.parameters["epsilon"]
        t = self.parameters["t"]
        delta = pow(10, -8)
        # b
        self.s_b = np.add(rho1 * self.s_b, (1 - rho1) * self.b_update)
        self.r_b = np.add(rho2 * self.r_b, (1 - rho2) * np.square(self.b_update))
        s_b_hat = self.s_b / (1 - pow(rho1, t))
        r_b_hat = self.r_b / (1 - pow(rho2, t))
        self.b = np.add(self.b, - epsilon * np.divide(s_b_hat, np.add(np.sqrt(r_b_hat), delta)))
        # w
        self.s_w = np.add(rho1 * self.s_w, (1 - rho1) * self.w_update)
        self.r_w = np.add(rho2 * self.r_w, (1 - rho2) * np.square(self.w_update))
        s_w_hat = self.s_w / (1 - pow(rho1, t))
        r_w_hat = self.r_w / (1 - pow(rho2, t))
        self.w = np.add(self.w, - epsilon * np.divide(s_w_hat, np.add(np.sqrt(r_w_hat), delta)))