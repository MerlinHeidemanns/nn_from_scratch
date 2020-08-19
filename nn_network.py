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

    def initialize(self):
        self.next.initialize()

    def forward(self, X, y):
        self.next.input = np.transpose(X)
        return(self.next.forward(y))

    def backward(self):
        self.next.backward()

    def update(self, parameters):
        self.next.update(parameters)

    def train_test_split(self, X, y, validation):
        n = X.shape[0]
        ind_train = np.random.choice(range(n), int(n * validation), replace=False)
        ind_test = list(set(range(n)) - set(ind_train))
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

    def train(self, X, y, epochs, minibatch, validation = 0.1, parameters = dict()):
        X_train, y_train, X_test, y_test = self.train_test_split(X, y, validation)
        self.parameters = parameters
        for i in range(epochs):
            print(i)
            j = 0
            while j < X_train.shape[0]:
                training_error = self.forward(X_train[j:j + minibatch], y_train[j:j + minibatch])
                self.backward()
                self.update(self.parameters)
                test_error     = self.get_accuracy(X_test, y_test)
                print(training_error)
                print(test_error)
                self.training_error.append(training_error)
                self.test_error.append(test_error)
                j += minibatch

