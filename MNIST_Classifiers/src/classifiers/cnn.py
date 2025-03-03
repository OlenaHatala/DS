from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Input

class ConvolutionalNeuralNetwork():
    def __init__(self, shape=(28,28,1), filters=32, 
                 kernel_size=(3,3), dropout_rate=0.5, 
                 hidden_units=500, optimizer='adam', 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'], **kwargs):
        """
        Initializes the Convolutional Neural Network (CNN) model with the specified architecture.

        Args:
            shape (tuple): Shape of the input data, typically (height, width, channels), default (28,28,1) for MNIST.
            filters (int): The number of filters in the convolutional layers, default is 32.
            kernel_size (tuple): Size of the convolutional kernel, default is (3,3).
            dropout_rate (float): The rate of dropout applied after the convolutional layers to prevent overfitting, default is 0.5.
            hidden_units (int): Number of units in the fully connected hidden layer, default is 500.
            optimizer (str): Optimizer to use during training, default is 'adam'.
            loss (str): The loss function to minimize, default is 'sparse_categorical_crossentropy'.
            metrics (list): List of metrics to evaluate during training, default is ['accuracy'].
            **kwargs: Additional arguments that can be passed to the constructor.
        """
        self.model = Sequential([
            Input(shape=shape),
            Conv2D(filters, kernel_size, activation='relu'),  
            MaxPool2D((2,2)),
            Conv2D(48, kernel_size, activation='relu'),
            MaxPool2D((2,2)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(hidden_units, activation='relu'),
            Dense(10, activation='softmax') 
        ])
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


    def train(self, x_train, y_train, epochs=5, batch_size=32, verbose=0, validation_split=0, **kwargs):
        """
        Trains the CNN model on the provided training data.

        Args:
            x_train (array-like): The input data for training (images).
            y_train (array-like): The corresponding labels for the training data.
            epochs (int): Number of training epochs, default is 5.
            batch_size (int): Number of samples per gradient update, default is 32.
            verbose (int): Verbosity mode (0 = silent, 1 = progress bar, 2 = one line per epoch), default is 0.
            validation_split (float): Fraction of the training data to be used as validation data, default is 0 (no validation).
            **kwargs: Additional parameters for the `fit` method.
        """
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=validation_split)
    
    def predict(self, x_test):
        """
        Makes predictions on the test data using the trained CNN model.

        Args:
            x_test (array-like): The input data for prediction (test images).

        Returns:
            array: Predicted class labels for the test data.
        """
        return self.model.predict(x_test).argmax(axis=1)
    
    def save(self, filepath):
        """
        Saves the trained CNN model to a file at the specified path.

        Args:
            filepath (str): The path where the model should be saved.
        """
        self.model.save(filepath)
        print(f"CNN model saved to {filepath}")

    def load(self, filepath):
        """
        Loads a pre-trained CNN model from a file at the specified path.

        Args:
            filepath (str): The path from which to load the model.
        """
        self.model = load_model(filepath)  # Load the model from file
        print(f"CNN model loaded from {filepath}")

