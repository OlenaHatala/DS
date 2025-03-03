from models.ner_model import NerModel

class NerInference:
    """
    A class for performing Named Entity Recognition (NER) inference using a pre-trained model.

    This class loads a pre-trained NER model, processes input text, and predicts named entities.

    Attributes:
        model (NerModel): An instance of the NER model used for entity recognition.
    """

    def __init__(self, model_weights="../src/weights/ner_model.h5"):
        """
        Initializes the NerInference class by loading a pre-trained model with specified weights.

        Args:
            model_weights (str): The file path to the saved model weights.
        """
        self.model = NerModel()
        self.model.load_weights(model_weights)

    def predict(self, text):
        """
        Performs Named Entity Recognition (NER) on the given text.

        Args:
            text (str): The input text containing potential named entities.

        Returns:
            set: A set of detected named entities.
        """
        detected_entities = self.model.predict(text)  
        return detected_entities

