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


    def _compute_a(self):
        if self.optimization == "nesterov_momentum" or self.optimization == "rmsprop_nesterov":
            alpha_momentum = self.parameters["alpha_momentum"]
            self.a = np.add(np.add(self.b, alpha_momentum * self.v_b),
                            np.matmul(np.add(self.w, alpha_momentum * self.v_w), self.input))
        else:
            self.a = np.add(self.b, np.matmul(self.w, self.input))

    def _compute_h(self):
        pass

    def forward(self, y=None, parameters=None):
        self.parameters = parameters
        self._compute_a()
        self.batch_norm()
        self._compute_h()
        self.next.forward(y=y, parameters=self.parameters)

    def backward(self):
        self.next.backward()

    def predict(self):
        self.a = np.add(self.b, np.matmul(self.w, self.input))
        self._compute_h()
        return (self.next.predict())

    def get_accuracy(self, y):
        self.a = np.add(self.b, np.matmul(self.w, self.input))
        self._compute_h()
        return (self.next.get_accuracy(y))

    def compute_update(self):
        self.n = self.input.shape[1]
        self.b_update = 1 / self.n * np.sum(self.delta, axis=1, keepdims=True)
        self.w_update = 1 / self.n * np.matmul(self.delta, np.transpose(self.input))

    def batch_norm(self):
        self.mu = np.mean(self.a, axis = 1)
        self.sigma2 = np.mean(np.square(np.transpose(self.a) - self.mu), axis = 0)
        self.a_transformed = np.transpose(np.divide(np.transpose(self.a) - self.mu, np.sqrt(self.sigma2 + 0.0001) ))
        self.a_shifted = self.beta + np.multiply(self.gamma, self.a_transformed)
        print(self.a_shifted.shape)

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


class ReLUHiddenLayer(HiddenLayer):

    def __init__(self, d):
        super().__init__(d=d)

    def _relu(self, x):
        return max([0, x])

    def _grelu(self, x):
        return np.where(x > 0, 1, 0)

    def _compute_h(self):
        _vrelu = np.vectorize(self._relu)
        self.next.input = _vrelu(self.a)

    def backward(self):
        super().backward()
        self.delta = np.multiply(self._grelu(self.a), self.next.gradient)
        self.gradient = np.matmul(np.transpose(self.w), self.delta)


class SigmoidHiddenLayer(HiddenLayer):

    def __init__(self, d):
        super().__init__(d=d)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _gsigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _compute_h(self):
        _vsigmoid = np.vectorize(self._sigmoid)
        self.next.input = _vsigmoid(self.a)

    def backward(self):
        super().backward()
        _vgsigmoid = np.vectorize(self._gsigmoid)
        self.delta = np.multiply(_vgsigmoid(self.a), self.next.gradient)
        self.gradient = np.matmul(np.transpose(self.w), self.delta)
