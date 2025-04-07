# Named Entity Recognition + Image Classification
---

## Overview
This project focuses on building an ML pipeline that combines Named Entity Recognition (NER) with Image Classification to verify whether a given text description correctly identifies an animal in an image.

## Task Description
1. The user provides a text input, e.g., _"There is a cow in the picture."_, and an image that potentially contains an animal.
2. The pipeline will extract the animal name from the text using an **NER model**.
3. An **image classification model** will then determine the actual animal in the image.
4. The pipeline will compare the extracted entity with the predicted animal in the image and return a boolean value:
   - `True` if the animal in the text matches the one in the image.
   - `False` otherwise.

## Datasets
- **Image Classification Dataset:** [Animals-10 Dataset](https://www.kaggle.com/datasets/viratkothari/animal10/data). The dataset contains labeled images of 10 animal classes used to train the classification model.
  • The training process is available in the `notebooks/ner_training.ipynb` file.

- **Text Dataset:** The text dataset is generated manually to include various ways of describing images containing animals.
  • The training process is available in the `notebooks/image_model_training.ipynb` file.

## Dataset Organization
- The image classification dataset is stored under datasets/images/Animals-10/, and divided into training (train/) and testing (test/) folders.
 • Inside test/, images are categorized into separate folders by class (e.g., butterfly/, cat/, dog/, etc.).
- The NER dataset is located in datasets/text/, containing:
 • ner_dataset.json: The text dataset for named entity recognition.
 • generate_dataset.py: A script to generate additional text-based datasets.

## Weights
NER Model and Image Model were trained, and the weights are saved by link: 
https://drive.google.com/drive/folders/1aQAjrUjYTYbd8pM18JiAyTRG5Z0y46QB?usp=drive_link
because they could not be uploaded to GitHub due to their large size. Just save them in the folder `weights`.
Alternatively, models can be trained using the provided scripts.

## Model Training
- **NER Model**: A transformer-based model trained to recognize animal names in text.  
  • The training process is available in the `notebooks/ner_training.ipynb` file.
- **Image Classification Model**: A deep learning model trained to classify images from the Animals-10 dataset.  
  • The training process is available in the `notebooks/image_model_training.ipynb` file.


## Project Structure
```
|-- datasets/
|   |-- images/
|   |   |-- Animals-10/  # Animal images dataset
|   |   |-- train/  # Training images (organized by class)
|   |   |-- test/  # Testing images (organized by class)
|   |       |-- butterfly/
|   |       |-- cat/
|   |       |-- chicken/
|   |       |-- cow/
|   |       |-- dog/
|   |       |-- elephant/
|   |       |-- horse/
|   |       |-- sheep/
|   |       |-- spider/
|   |       |-- squirrel/
|   |
|   |-- text/  # Generated text dataset for NER
|       |-- generate_dataset.py  # Script for generating NER dataset
|       |-- ner_dataset.json  # JSON file with text-based dataset

|-- models/
|   |-- image_classification_model.py  # Image classification model
|   |-- ner_model.py  # NER model
|   |-- model_interface.py  # Model inference interfaces

|-- notebooks/
|   |-- image_model_training.ipynb  # Training notebook for image classification
|   |-- ner_training.ipynb  # Training notebook for NER model
|   |-- pipeline_using.ipynb  # Demo and EDA notebook

|-- src/
|   |-- inferences/
|   |   |-- image_inference.py  # Image classification inference
|   |   |-- ner_inference.py  # NER inference
|   |-- weights/  # Trained model weights 

|-- pipeline.py  # Main script combining NER and Image Classification
|-- requirements.txt  # Dependencies
|-- README.md  # Project documentation

```

## Pipeline Execution
The `pipeline_using.ipynb` notebook demonstrates how to run the verification pipeline step by step, integrating both models:
1. **NER Model** extracts the animal name from the given text.
2. **Image Classification Model** predicts the animal in the provided image.
3. The pipeline compares both outputs and verifies if they match.

---

### Running the Pipeline in Jupyter Notebook
To execute the pipeline, **open and run** the following notebook:

📌 **Notebook:** `notebooks/pipeline_using.ipynb`

This notebook provides:
- ✅ Step-by-step execution of the NER and image classification models.  
- ✅ Clear visualization of detected entities and predictions.  
- ✅ Final verification result.

---

### Expected Output
When running the pipeline on a text and image containing a cow, the output might look like this:

```plaintext
NER Detected: {'cow'}
Image Predicted: cow

Verification Result: True

```

## Dependencies
To install required dependencies:
```bash
pip install -r requirements.txt
```
or
```bash
conda install --yes --file requirements.txt
```

