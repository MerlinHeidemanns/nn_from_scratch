import numpy as np

class Module():

    def __init__(self):
        self.prev   = None
        self.next   = None

    def __call__(self, other = None):
        if isinstance(other, Module):
            other.k = self.d
            self.next = other
            other.prev = self
        #else:
        #    self.output = other

    def forward(self):
        pass

    def backward(self):
        pass

class InputLayer(Module):

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.h = None

class HiddenLayer(Module):

    def __init__(self, d, activation):
        super().__init__()
        self.d = d
        self.h = None
        self.b = None
        self.w = None
        self.delta = None
        self.update_bias = None
        self.update_weights = None
        self.activation = activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return max([0, x])

    def grelu(self, x):
        return np.where(x > 0, 1, 0)

    def compute_a(self):
        self.a = np.add(self.b, np.matmul(self.w, self.prev.h))

    def compute_h(self):
        if self.activation == "linear":
            self.h = self.a
        elif self.activation == "ReLU":
            vrelu = np.vectorize(self.relu)
            self.h = vrelu(self.a)
        elif self.activation == "sigmoid":
            vsigmoid = np.vectorize(self.sigmoid)
            self.h = vsigmoid(self.a)

    def forward(self):
        self.n = self.prev.h.shape[1]
        self.compute_a()
        self.compute_h()

    def backprop(self):
        if self.activation == "linear":
            self.delta = self.next.gradient
        elif self.activation == "ReLU":
            self.delta = np.multiply(self.grelu(self.a), self.next.gradient)
        elif self.activation == "sigmoid":
            self.delta = np.multiply(self.gsigmoid(self.a), self.next.gradient)
        self.gradient = np.matmul(np.transpose(self.w), self.delta)

    def update(self, optimizer, regularizer = None, epsilon = 0.5, epsilon = None, rho1 = 0.9, rho2 = 0.99):
        try:
            self.t
        except:
            self.t = 0
        self.t += 1
        self.update_bias    = 1/self.n * np.sum(self.delta, axis = 1, keepdims=True)
        self.update_weights = 1/self.n * np.matmul(self.delta, np.transpose(self.prev.h))
        if optimizer == "sgd":
            self.update_sgd()
        elif optimizer == "adam":
            self.update_adam(t = self.t, rho1 = rho1, rho2 = rho2)
        if regularizer == "L2":
            self.regularize_update_L2(epsilon = epsilon, epsilon = epsilon)
        else:
            self.update_weights = epsilon * self.update_weights
        self.b = np.subtract(self.b, epsilon * self.update_bias)
        self.w = np.subtract(self.w, self.update_weights)

    def regularize_update_L2(self, epsilon, epsilon):
        self.update_weights = epsilon * np.add(epsilon * self.w, self.update_weights)

    def update_sgd(self):
        pass

    def update_momentum(self, epsilon, epsilon):
        try:
            self.velocity
        except:
            self.velocity = np.zeros(shape=self.w.shape)
            self.velocity = epsilon * self.velocity - epsilon * self.update_weights
            self.update_weights = - self.velocity

    def update_adam(self, t, rho1, rho2):
        if t == 1:
            self.s_w = np.zeros(shape = self.w.shape)
            self.r_w = np.zeros(shape = self.w.shape)
        self.s_w = np.add(rho1 * self.s_w, np.multiply(1 - rho1, self.update_weights))
        self.r_w = np.add(rho2 * self.r_w, np.multiply(1 - rho2, np.square(self.update_weights)))
        s_w_bar  = np.divide(self.s_w, (1 - pow(rho1, t)))
        r_w_bar  = np.divide(self.r_w, (1 - pow(rho2, t)))
        self.update_weights = np.divide(s_w_bar, np.add(np.sqrt(r_w_bar), 0.001))



class HiddenLayerLinear(HiddenLayer):

    def __init__(self, d):
        super().__init__(d = d)

    def backprop(self):
        self.delta    = self.next.gradient
        self.gradient = np.matmul(np.transpose(self.w), self.delta)

class HiddenLayerSigmoid(HiddenLayer):

    def __init__(self, d):
        super().__init__(d = d)

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

class HiddenLayerReLU(HiddenLayer):

    def __init__(self, d):
        super().__init__(d = d)

    def relu(self, x):
        return max([0, x])

    def grelu(self, x):
        return np.where(x > 0, 1, 0)

    def compute_h(self):
        vrelu = np.vectorize(self.relu)
        self.h = vrelu(self.a)

    def backprop(self):
        self.delta     = np.multiply(self.grelu(self.a), self.next.gradient)
        self.gradient = np.matmul(np.transpose(self.w), self.delta)

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


class CELossLayer(LossLayer):

    def __init__(self):
        pass

    def forward(self):
        # mean(y != y_hat
        #print(np.column_stack((np.transpose(self.prev.h), np.transpose(self.y))))
        self.loss = np.mean(np.where(self.prev.h > 0.5, 1, 0) != self.y)

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
            layer.w = np.random.uniform(low = -0.2, high = 0.2, size = (layer.d, layer.k))
            layer.b = np.random.uniform(low = -0.2, high = 0.2, size = (layer.d, 1))
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
            return None
        elif layer is None:
            layer = self.input_layer
        else:
            layer.forward()
            layer = layer.next
        self.forward(layer = layer)

    def update(self,layer = None):
        if isinstance(layer, LossLayer):
            return None
        elif layer is None:
            layer = self.input_layer.next
        else:
            layer.update(optimizer = self.optimizer, regularizer = self.regularizer,
                         epsilon = self.epsilon, epsilon = self.epsilon, rho1 =self.rho1, rho2 = self.rho2)
            layer = layer.next
        self.update(layer = layer)

    #def backprop(self, ):
    #    layer.next

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

    def set_optimizer(self, optimizer):
        if optimizer is None:
            self.optimizer = None
        else:
            options = ["sgd", "adam"]
            if optimizer in options:
                self.optimizer = optimizer
            else:
                print("This optimizer either does not exist or hasn't been implemented yet. Please choose one of:\n")
                for i in options:
                    print(i, "\n")
                raise NotImplementedError("")

    def set_regularizer(self, regularizer):
        if regularizer is None:
            self.regularizer = None
        else:
            options = ["L2"]
            if regularizer in options:
                self.regularizer = regularizer
            else:
                print("This regularizer either does not exist or hasn't been implemented yet. Please choose one of:\n")
                for i in options:
                    print(i, "\n")
                raise NotImplementedError("")

    def set_epsilon_adapt(self, adaptation, epsilon_delta):
        options = ["decay"]
        if adaptation in options:
            self.adaptation = adaptation
            self.epsilon_delta = epsilon_delta
        else:
            self.adaptation = None

    def adjust_epsilon(self, t):
        if self.adaptation is "decay":
            self.epsilon *= epsilon
        elif self.adaptation is "linear":
            self.epsilon = (1 - epsilon) +
        elif self.adaptation is None:
            pass

    def training(self, X, y, epochs, minibatch,
                 validation,
                 optimizer = "sgd", regularizer = None, adaptation = None, epsilon_delta = 0.99,
                 epsilon = 1, epsilon = 0.5,
                 rho1 = 0.9, rho2 = 0.99):
        X_train, y_train, X_test, y_test = self._train_test_split(X = X, y = y, validation = validation)
        self.set_optimizer(optimizer=optimizer)
        self.set_regularizer(regularizer=regularizer)
        self.set_epsilon_adapt(adaptation, epsilon_delta)
        self.epsilon = epsilon
        self.rho1 = rho1
        self.rho2 = rho2
        self.initialize()
        self.training_loss = []
        self.test_loss     = []
        p = np.random.permutation(X_train.shape[0])
        for i in range(epochs):
            print("Starting epoch ", i + 1)
            X_train, y_train = X_train[p], y_train[p]
            flag = True
            j    = -1
            while flag:
                j += 1
                start, end, flag = self._minibatch_subset(j, minibatch, X.shape[0])
                self.set_data(X[start:end], y[start:end])
                self.forward()
                self.backprop()
                self.update()
            # current output
            training_loss = self.top_layer.loss
            test_loss = self._return_loss(X_test, y_test)
            print("Loss on training set: ", round(training_loss, 4))
            print("Loss on test set:     ", round(test_loss, 4))
            self.training_loss.append(training_loss)
            self.test_loss.append(test_loss)
            # adjust epsilon
            self.adjust_epsilon(i)


    def predict(self, X):
        self.set_data(X, y = np.zeros(shape=(X.shape[0], 1)))
        self.forward()
        return self.top_layer.prev.h

    def get_training(self):
        out_dict = dict()
        out_dict["Training loss"] = self.training_loss
        out_dict["Test loss"]     = self.test_loss
        return out_dict

# python3 feedforward_neuralnet.py