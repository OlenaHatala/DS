from inferences.ner_inference import NerInference
from inferences.image_inference import ImageInference

class VerificationPipeline:
    """
    A pipeline for verifying if the detected named entity from text matches the predicted class from an image.

    This class integrates two models:
        - A Named Entity Recognition (NER) model for extracting animal names from text.
        - An image classification model for recognizing animals in images.
    
    Attributes:
        ner_model (NerInference): Instance of the NER model for text analysis.
        image_model (ImageInference): Instance of the image classification model for image analysis.
    """

    def __init__(self):
        """
        Initializes the verification pipeline by loading both the NER and image classification models.
        """
        self.ner_model = NerInference()
        self.image_model = ImageInference()


    def clean_ner_output(self, detected_entities):
        """
        Cleans and normalizes NER output labels.

        Converts detected entity labels (e.g., 'B-COW') into a lowercase format ('cow')
        for consistent comparison with image classification results.

        Args:
            detected_entities (set): A set of detected entity labels from the NER model.

        Returns:
            set: A set of normalized entity names.
        """
        return {entity.replace("B-", "").lower() for entity in detected_entities}


    def verify(self, text, img_path):
        """
        Verifies whether the predicted animal from the image matches any detected entities in the text.

        Args:
            text (str): Input text containing potential animal mentions.
            img_path (str): Path to the image file.

        Returns:
            bool: True if the image classification matches an entity detected in the text, False otherwise.
        """
        detected_entities = self.ner_model.predict(text)
        cleaned_entities = self.clean_ner_output(detected_entities)

        predicted_animal = self.image_model.predict(img_path).lower()

        print(f"NER Detected: {cleaned_entities}")
        print(f"Image Predicted: {predicted_animal}")

        is_correct = predicted_animal in cleaned_entities
        return is_correct

# # CLI interface for testing
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Verify if the image matches the text")
#     parser.add_argument("--text", type=str, required=True, help="Input sentence describing the image")
#     parser.add_argument("--image", type=str, required=True, help="Path to the image file")

#     args = parser.parse_args()
    
#     pipeline = VerificationPipeline()
#     result = pipeline.verify(args.text, args.image)
    
#     print(f"Verification Result: {'Correct' if result else 'Incorrect'}")
