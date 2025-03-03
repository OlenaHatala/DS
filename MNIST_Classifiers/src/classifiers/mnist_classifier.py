from interfaces.mnist_classifier_interface import MnistClassifierInterface
from classifiers.cnn import ConvolutionalNeuralNetwork
from classifiers.ffnn import FeedForwardConvolutionalNetwork
from classifiers.random_forest import RandomForest
import numpy as np

class MnistClassifier(MnistClassifierInterface):
    """
    Class that provides an interface to train and test models 
    for the MNIST dataset classification task. It can utilize different algorithms 
    including Convolutional Neural Networks (CNN), Feed-Forward Neural Networks (FFNN), 
    and Random Forest (RF).
    """

    def __init__(self, algorithm, **kwargs):
        """
        Initializes the classifier model based on the chosen algorithm.
        
        Args:
            algorithm (str): The algorithm to use for classification. Options are:
                              'cnn' for Convolutional Neural Network,
                              'rf' for Random Forest,
                              'nn' for Feed-Forward Neural Network.
            **kwargs: Additional parameters that might be passed to the model constructors.
        
        Raises:
            ValueError: If the specified algorithm is not recognized.
        """
        if algorithm == "cnn":
            self.model = ConvolutionalNeuralNetwork(**kwargs)
        elif algorithm == "rf":
            self.model = RandomForest(**kwargs)
        elif algorithm == "nn":
            self.model = FeedForwardConvolutionalNetwork(**kwargs)
        else: 
            raise ValueError("Invalid algorithm.")
        
    def train(self, x_train, y_train, **kwargs):
        """
        Trains the selected model using the provided training data.

        Args:
            x_train (array-like): The input data for training.
            y_train (array-like): The corresponding labels for the training data.
            **kwargs: Additional parameters to be passed to the model's training method.
        """
        self.model.train(x_train, y_train, **kwargs)

    def predict(self, x_test):
        """
        Makes predictions on the test data using the trained model.
        
        Args:
            x_test (array-like): The input data for prediction.

        Returns:
            array: The predicted labels for the input test data.
        """
        return self.model.predict(x_test)
    
    def accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy of the model by comparing the true labels with 
        the predicted labels.

        Args:
            y_true (array-like): The true labels.
            y_pred (array-like): The predicted labels.

        Returns:
            float: The accuracy of the model, computed as the ratio of correct predictions.
        """
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def save(self, filepath):
        """
        Saves the trained model to the specified file path.
        
        Args:
            filepath (str): The path where the model should be saved.
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Loads a pre-trained model from the specified file path.
        
        Args:
            filepath (str): The path from which to load the model.
        """
        self.model.load(filepath)
