import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense, GlobalAvgPool2D as GAP, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_interface import ModelInterface 

class ImageClassifier(ModelInterface):
    """
    ImageClassifier is a deep learning model for image classification based on ResNet152V2.
    The model is fine-tuned for a 10-class classification task, utilizing transfer learning 
    from a pre-trained ResNet152V2 model.
    
    Attributes:
        name (str): The name of the model.
        model (Sequential): The TensorFlow Keras model architecture.
    """
    def __init__(self):
        """
        Initializes the ImageClassifier model.
        
        The model uses a pre-trained ResNet152V2 as a feature extractor with a custom 
        classification head consisting of a Global Average Pooling layer, a dense 
        fully connected layer with ReLU activation, and an output layer with softmax activation.
        """
        self.name = "ResNet152V2"
        base_model = ResNet152V2(include_top=False,
                                 input_shape=(256,256,3),
                                 weights='imagenet')

        self.model = Sequential([
            base_model,
            GAP(),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')
        ], name=self.name)

        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train(self, train_data, valid_data, epochs=15, batch_size=32):
        """
        Trains the model on the given dataset.
        
        Args:
            train_data: Training dataset (can be a TensorFlow dataset generator).
            valid_data: Validation dataset.
            epochs (int): Number of training epochs. Default is 15.
            batch_size (int): Size of training batches. Default is 32.
        
        The training process includes early stopping and model checkpointing to save the best model.
        """
        cbs = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ModelCheckpoint(self.name + '.h5', save_best_only=True)
        ]
        self.model.fit(train_data, validation_data=valid_data, epochs=epochs, callbacks=cbs)

    def predict(self, x_test):
        """
        Generates predictions for the given test dataset.
        
        Args:
            x_test: Input images for prediction.
        
        Returns:
            numpy.ndarray: Predicted class probabilities.
        """
        return self.model.predict(x_test)

    def save_weights(self, path):
        """
        Saves the model weights to the specified path.
        
        Args:
            path (str): File path to save the weights.
        """
        self.model.save_weights(path)

    def load_weights(self, path):
        """
        Loads model weights from the specified path.
        
        Args:
            path (str): File path to load the weights from.
        """
        self.model.load_weights(path)