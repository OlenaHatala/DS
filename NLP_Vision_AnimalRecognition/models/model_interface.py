from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """
    ModelInterface is an abstract base class that defines a common interface 
    for machine learning models.
    
    Subclasses must implement the train, predict, save_weights, and load_weights methods.
    """
    
    @abstractmethod
    def train(self, **kwargs):
        """
        Trains the model using provided data and parameters.
        
        Args:
            **kwargs: Arbitrary keyword arguments for flexible training configurations.
            This allows different models to accept different sets of parameters 
            (e.g., training data, validation data, hyperparameters).
        """
        pass

    @abstractmethod
    def predict(self, x_test):
        """
        Generates predictions based on the input test data.
        
        Args:
            x_test: The input data for which predictions are required.
        
        Returns:
            Model predictions.
        """
        pass

    @abstractmethod
    def save_weights(self, path):
        """
        Saves the model weights to the specified file path.
        
        Args:
            path (str): The file path where weights should be stored.
        """
        pass

    @abstractmethod
    def load_weights(self, path):
        """
        Loads the model weights from the specified file path.
        
        Args:
            path (str): The file path from which to load weights.
        """
        pass