import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
from collections import Counter
from model_interface import ModelInterface

class NerModel(ModelInterface):
    """
    NerModel is a Named Entity Recognition (NER) model based on a transformer architecture.
    The model is fine-tuned to recognize specific animal-related entities in text using 
    token classification.
    
    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for processing input text.
        model (TFAutoModelForTokenClassification): Pre-trained transformer model for token classification.
        label_map (dict): Mapping of entity labels to their respective integer indices.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initializes the NER model using a pre-trained transformer model.
        
        Args:
            model_name (str): The name of the pre-trained model to use. Default is "distilbert-base-uncased".
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForTokenClassification.from_pretrained(model_name, num_labels=11)

        for layer in self.model.layers[:-1]:
            layer.trainable = False
        
        self.label_map = {
            "O": 0,
            "B-BUTTERFLY": 1, "B-CAT": 2, "B-CHICKEN": 3, "B-COW": 4, "B-DOG": 5,
            "B-ELEPHANT": 6, "B-HORSE": 7, "B-SHEEP": 8, "B-SQUIRREL": 9, "B-ZEBRA": 10
        }

    def process_data(self, sentences: list, labels: list):
        """
        Tokenizes sentences while maintaining label alignment.

        Args:
            sentences (list): List of tokenized words per sentence.
            labels (list): Corresponding labels for each tokenized word.
        
        Returns:
            tuple: Tokenized input tensors and aligned label arrays.
        """
        encoded_data = self.tokenizer(
            [" ".join(sentence) for sentence in sentences],
            padding=True, truncation=True, return_tensors="tf", max_length=128
        )
        
        aligned_labels = []
        for idx, label_seq in enumerate(labels):
            word_ids = encoded_data.word_ids(batch_index=idx)
            aligned_seq = []

            prev_word_id = None
            for word_id in word_ids:
                if word_id is None or word_id >= len(label_seq):
                    aligned_seq.append(-100)  
                elif word_id != prev_word_id:
                    aligned_seq.append(self.label_map.get(label_seq[word_id], 0))
                else:
                    aligned_seq.append(-100)  
                
                prev_word_id = word_id

            aligned_labels.append(aligned_seq)

        return encoded_data, np.array(aligned_labels)
    
    def train(self, x_train, y_train, x_valid, y_valid, epochs=10, batch_size=32):
        """
        Trains the NER model on the provided dataset.
        
        Args:
            x_train: Training sentences.
            y_train: Corresponding labels for training data.
            x_valid: Validation sentences.
            y_valid: Corresponding labels for validation data.
            epochs (int): Number of training epochs. Default is 10.
            batch_size (int): Batch size for training. Default is 32.
        """
        train_inputs, train_labels = self.process_data(x_train, y_train)
        valid_inputs, valid_labels = self.process_data(x_valid, y_valid)
    
        unique_entities, counts = np.unique(train_labels, return_counts=True)
        print(f"Detected unique entity labels: {dict(zip(unique_entities, counts))}")
    
        if len(unique_entities) == 1 and unique_entities[0] == 0:
            raise ValueError("Dataset only contains 'O' labels! Check label mapping.")
        
        train_labels = np.where(train_labels == -100, 0, train_labels)
        valid_labels = np.where(valid_labels == -100, 0, valid_labels)
    
        entity_frequencies = Counter(train_labels.flatten())
        total_samples = sum(entity_frequencies.values())
        loss_weights = {label: total_samples / (len(entity_frequencies) * count) for label, count in entity_frequencies.items()}
    
        sample_weights = np.vectorize(loss_weights.get)(train_labels)
    
        training_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(train_inputs), train_labels, sample_weights)
        ).shuffle(1000).batch(batch_size)
    
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (dict(valid_inputs), valid_labels)
        ).batch(batch_size)
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
    
        self.model.fit(training_dataset, validation_data=validation_dataset, epochs=epochs)

    def predict(self, x_test):
        """
        Identifies named entities related to animals in a given text.
        
        Args:
            x_test: Input text containing potential animal names.
        
        Returns:
            set: Detected animal-related entities.
        """
        encoded_text = self.tokenizer(x_test, return_tensors="tf")
        logits = self.model(**encoded_text).logits
        predictions = tf.argmax(logits, axis=-1).numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(encoded_text["input_ids"].numpy()[0])
        detected_animals = set()
        for token, label in zip(tokens, predictions[0]):
            entity_name = list(self.label_map.keys())[label]
            if entity_name != "O":
                detected_animals.add(entity_name)
        return detected_animals

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
