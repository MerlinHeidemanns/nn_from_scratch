class Module():

    def __init__(self):
        self.next = None
        self.input = None


    def __call__(self, other):
        if isinstance(other, Module):
            self.next = other

    def forward(self, y = None, parameters = None):
        pass

    def backward(self):
        pass

    def initialize(self, optimization, initialization, regularization):
        self.next.initialize(optimization, initialization, regularization)

    def update(self, parameters = None):
        self.next.update()

    def predict(self):
        pass

    def show_architecture(self):
        print(self.__class__.__name__)
        self.next.show_architecture()