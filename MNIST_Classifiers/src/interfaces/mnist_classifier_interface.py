from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_test):
        pass