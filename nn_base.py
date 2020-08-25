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

    def initialize(self, optimization, initialization):
        pass

    def update(self, parameters = None):
        pass

    def predict(self):
        pass