import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from models.image_classification_model import ImageClassifier

class ImageInference:
    """
    A class for performing image classification inference using a pre-trained model.

    This class loads a pre-trained image classification model, preprocesses input images, 
    and makes predictions to identify the class of an image.

    Attributes:
        model (ImageClassifier): An instance of the image classification model.
        class_labels (list): A list of class labels corresponding to the model's output.
    """

    def __init__(self, model_weights="../src/weights/image_model.h5"):
        """
        Initializes the ImageInference class by loading a pre-trained model with specified weights.

        Args:
            model_weights (str): The file path to the saved model weights.
        """
        self.model = ImageClassifier()
        self.model.load_weights(model_weights)
        self.class_labels = ["butterfly", "cat", "chicken", "cow", "dog", 
                             "elephant", "horse", "sheep", "spider", "squirrel"]

    def preprocess_image(self, img_path):
        """
        Loads and preprocesses an image for model inference.

        Args:
            img_path (str): The file path to the image.

        Returns:
            numpy.ndarray: A preprocessed image array suitable for model prediction.
        """
        img = image.load_img(img_path, target_size=(256, 256)) 
        img_array = image.img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  
        return img_array

    def predict(self, img_path):
        """
        Performs inference on the given image and returns the predicted class label.

        Args:
            img_path (str): The file path to the image.

        Returns:
            str: The predicted class label corresponding to the highest probability.
        """
        img_array = self.preprocess_image(img_path)
        predictions = self.model.predict(img_array) 
        predicted_label = self.class_labels[np.argmax(predictions)]  
        return predicted_label
