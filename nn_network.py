from nn_base import *
from nn_loss import *
from nn_hidden import *
from matplotlib import pyplot as plt

class Network(Module):

    def __init__(self):
        super().__init__()
        self.training_error = []
        self.test_error = []

    def __call__(self, other, d = None):
        if isinstance(other, Module):
            if self.next is None:
                self.next = other
                other.k = d
            else:
                self.next(other)

    def show_architecture(self):
        self.next.show_architecture()

    def initialize(self, optimization, initialization, regularization):
        self.parameters["t"] = 0
        self.next.initialize(optimization, initialization, regularization)

    def forward(self, X, y):
        self.next.input = np.transpose(X)
        self.next.forward(y = y, parameters=self.parameters)

    def backward(self):
        self.next.backward()

    def update(self):
        self.next.update()

    def train_test_split(self, X, y, validation):
        n = X.shape[0]
        ind_test = np.random.choice(range(n), int(n * validation), replace=False)
        ind_train = list(set(range(n)) - set(ind_test))
        return X[ind_train], y[ind_train], X[ind_test], y[ind_test]

    def get_accuracy(self, X, y):
        self.next.input = np.transpose(X)
        return(self.next.get_accuracy(y))

    def show_error(self):
        epochs = 1 + len(self.training_error)
        plt.plot(list(range(1, epochs)), self.training_error, color='blue', label="Training")
        plt.plot(list(range(1, epochs)), self.test_error, color='red', label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("1 - accuracy")
        plt.legend()
        plt.show()

    def adjust_parameters(self):
        self.parameters["t"] += 1
        if self.parameters["t"] % 20 == 0:
            self.parameters["epsilon"] *= self.parameters["epsilon_adjustment"]

    def train(self, X, y, epochs, minibatch, validation = 0.1, parameters = dict(),
              optimization   = "sgd",
              initialization = None,
              regularization = None):
        X_train, y_train, X_test, y_test = self.train_test_split(X, y, validation)
        self.parameters = parameters
        self.initialize(optimization, initialization, regularization)
        for i in range(1, 1+ epochs):
            print("Epoch: ", i)
            self.adjust_parameters()
            j = 0
            while j < X_train.shape[0]:
                self.forward(X_train[j:j + minibatch], y_train[j:j + minibatch])
                self.backward()
                self.update()
                j += minibatch
            training_error = self.get_accuracy(X_train, y_train)
            test_error     = self.get_accuracy(X_test, y_test)
            self.training_error.append(training_error)
            self.test_error.append(test_error)
            print("Training loss:", training_error)
            print("Test loss:    ", test_error)

    def predict(self, X):
        self.next.input = np.transpose(X)
        return(np.transpose(self.next.predict()))