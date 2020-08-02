import numpy as np
from matplotlib import pyplot as plt

class Module():

    def __init__(self):
        self.prev   = None
        self.next   = None

    def __call__(self, other = None):
        if isinstance(other, Module):
            other.k = self.d
            self.next = other
            other.prev = self
        else:
            self.output = other

    def forward(self):
        pass

    def backward(self):
        pass

class InputLayer(Module):

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.h = None

class HiddenLayerSigmoid(Module):

    def __init__(self, d, alpha):
        super().__init__()
        self.d = d
        self.h = None
        self.b = None
        self.w = None
        self.alpha = alpha

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def compute_h(self):
        vsigmoid = np.vectorize(self.sigmoid)
        self.h = vsigmoid(self.a)

    def compute_a(self):
        self.a = np.add(self.b, np.matmul(self.w, self.prev.h))

    def forward(self):
        self.n = self.prev.h.shape[1]
        self.compute_a()
        self.compute_h()

    def backprop(self):
        self.delta          = np.multiply(self.gsigmoid(self.a), self.next.gradient)
        self.gradient       = np.matmul(np.transpose(self.w), self.delta)
        self.update_bias    = 1/self.n * self.delta
        self.update_weights = 1/self.n * np.matmul(self.delta, np.transpose(self.prev.h))

    def update(self):
        self.b = np.subtract(self.b, self.alpha * self.update_bias)
        self.w = np.subtract(self.w, self.alpha * self.update_weights)

class HiddenLayer(Module):

    def __init__(self, d, alpha):
        super().__init__()
        self.d = d
        self.h = None
        self.b = None
        self.w = None
        self.alpha = alpha
        self.delta = None
        self.update_bias = None
        self.update_weights = None

    def compute_a(self):
        self.a = np.add(self.b, np.matmul(self.w, self.prev.h))

    def compute_h(self):
        self.h = self.a

    def update(self):
        self.update_bias    = 1/self.n * np.sum(self.delta, axis = 1, keepdims=True)
        self.update_weights = 1/self.n * np.matmul(self.delta, np.transpose(self.prev.h))
        self.b = np.subtract(self.b, self.alpha * self.update_bias)
        self.w = np.subtract(self.w, self.alpha * self.update_weights)

    def update_adam(self, t, rho1, rho2):
        self.update_bias    = 1/self.n * np.sum(self.delta, axis = 1, keepdims=True)
        self.update_weights = 1/self.n * np.matmul(self.delta, np.transpose(self.prev.h))
        if t == 1:
            self.s_w = np.zeros(shape = self.w.shape)
            self.s_b = np.zeros(shape = self.b.shape)
            self.r_w = np.zeros(shape = self.w.shape)
            self.r_b = np.zeros(shape = self.b.shape)
        self.s_w = np.add(rho1 * self.s_w, np.multiply(1 - rho1, self.update_weights))
        self.s_b = np.add(rho1 * self.s_b, np.multiply(1 - rho1, self.update_bias))
        self.r_w = np.add(rho2 * self.r_w, np.multiply(1 - rho2, np.square(self.update_weights)))
        self.r_b = np.add(rho2 * self.r_b, np.multiply(1 - rho2, np.square(self.update_bias)))
        s_w_bar  = np.divide(self.s_w, (1 - pow(rho1, t)))
        s_b_bar  = np.divide(self.s_b, (1 - pow(rho1, t)))
        r_w_bar  = np.divide(self.r_w, (1 - pow(rho1, t)))
        r_b_bar  = np.divide(self.r_b, (1 - pow(rho1, t)))
        self.w = np.subtract(self.w, self.alpha * np.divide(s_w_bar, np.add(np.sqrt(r_w_bar), 0.001)))
        self.b = np.subtract(self.b, self.alpha * np.divide(s_b_bar, np.add(np.sqrt(r_b_bar), 0.001)))

    def forward(self):
        self.n = self.prev.h.shape[1]
        self.compute_a()
        self.compute_h()

class HiddenLayerLinear(HiddenLayer):

    def __init__(self, d, alpha):
        super().__init__(d = d, alpha = alpha)

    def backprop(self):
        self.delta    = self.next.gradient
        self.gradient = np.matmul(np.transpose(self.w), self.delta)

class HiddenLayerSigmoid(HiddenLayer):

    def __init__(self, d, alpha):
        super().__init__(d = d, alpha = alpha)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def compute_h(self):
        vsigmoid = np.vectorize(self.sigmoid)
        self.h = vsigmoid(self.a)

    def backprop(self):
        self.delta          = np.multiply(self.gsigmoid(self.a), self.next.gradient)
        self.gradient       = np.matmul(np.transpose(self.w), self.delta)

class LossLayer(Module):

    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError("Implement forward functionality")

    def backprop(self):
        raise NotImplementedError("Implement backprop functionality")

class MSELossLayer(LossLayer):

    def __init__(self):
        pass

    def forward(self):
        # sum(xb - y)^2/n
        self.loss = np.square(np.subtract(self.prev.h, self.y)).mean()

    def backprop(self):
        self.gradient = np.subtract(self.prev.h, self.y)

class Network():

    def __init__(self):
        self.input_layer = None
        self.top_layer = None
        self.bias = []
        self.weights = []

    def initialize(self, layer = None):
        # d is the number of neurons of the layer
        # k is the number of inputs
        if isinstance(layer, LossLayer):
            return None
        elif layer is None:
            layer = self.input_layer.next
        else:
            layer.w = np.random.uniform(low = -0.1, high = 0.1, size = (layer.d, layer.k))
            layer.b = np.random.uniform(low = -0.1, high = 0.1, size = (layer.d, 1))
            layer = layer.next
        self.initialize(layer = layer)

    def set_input_layer(self, d):
        self.input_layer = InputLayer(d)
        self.top_layer = self.input_layer

    def add_layer(self, layer):
        self.top_layer(layer)
        self.top_layer = layer

    def set_data(self, X, y):
        self.input_layer.h = np.transpose(X)
        if isinstance(self.top_layer, LossLayer):
            self.top_layer.y = np.transpose(y)
        else:
            raise TypeError("Last layer is not a loss layer.")

    def forward(self, layer = None):
        if isinstance(layer, LossLayer):
            layer.forward()
            print(layer.prev.h)
            return None
        elif layer is None:
            layer = self.input_layer
        else:
            layer.forward()
            layer = layer.next
        self.forward(layer = layer)

    def update(self, t = None,layer = None):
        if isinstance(layer, LossLayer):
            return None
        elif layer is None:
            layer = self.input_layer.next
        else:
            layer.update_adam(t = t, rho1 = 0.99, rho2 = 0.999)
            layer = layer.next
        self.update(t = t, layer = layer)

    def backprop(self, layer = None):
        if layer is None:
            layer = self.top_layer
        elif isinstance(layer, InputLayer):
            return None
        else:
            layer.backprop()
            layer = layer.prev
        self.backprop(layer = layer)

    def _minibatch_subset(self, iteration, minibatch, length):
        start = iteration * minibatch
        end   = (iteration + 1) * minibatch + 1
        flag = True
        if end > length:
            end = length
            flag = False
        return start, end, flag

    def _train_test_split(self, X, y, validation):
        n = X.shape[0]
        ind_train = np.random.choice(range(n), int(n * validation), replace=False)
        ind_test = list(set(range(n)) - set(ind_train))
        X_train, y_train = X[ind_train], y[ind_train]
        X_test, y_test = X[ind_test], y[ind_test]
        return X_train, y_train, X_test, y_test

    def _return_loss(self, X, y):
        self.set_data(X, y)
        self.forward()
        return self.top_layer.loss

    def training(self, X, y, epochs, minibatch, validation):
        X_train, y_train, X_test, y_test = self._train_test_split(X = X, y = y, validation = validation)
        self.initialize()
        for i in range(epochs):
            print("Starting epoch ", i + 1)
            p = np.random.permutation(X_train.shape[0])
            X_train, y_train = X_train[p], y_train[p]
            flag = True
            j    = -1
            while flag:
                j += 1
                start, end, flag = self._minibatch_subset(j, minibatch, X.shape[0])
                self.set_data(X[start:end], y[start:end])
                self.forward()
                self.backprop()
                self.update(t = j + 1)
            print("Loss on test set:", self._return_loss(X_test, y_test))

    def predict(self, X):
        self.set_data(X, y = np.zeros(shape=(X.shape[0], 1)))
        self.forward()
        return self.top_layer.prev.h

if __name__ == "__main__":
    K = 1
    N = 300
    X = np.random.uniform(-10, 10, size = (N, K))
    b = np.ones(shape=(1,1))
    y = np.sign(np.matmul(X, b) + 3)
    plt.scatter(X, y)
    plt.show()
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

# python3 feedforward_neuralnet.py