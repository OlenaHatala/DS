# Multimodal Classification Projects

This directory contains two separate machine learning projects, each focusing on different classification tasks. Each project has its own detailed description in its respective `README.md` file and includes a separate `requirements.txt` file for dependencies.

---

## **1. MNIST Classifiers**
This project focuses on **digit classification** using different neural network architectures. The models are trained on the **MNIST dataset** to recognize handwritten digits from 0 to 9.

### **Main Components:**
- **notebooks/** – Jupyter notebooks for training and evaluating different classifiers.
- **src/**  
  - `classifiers/` – Contains implementations of various models.  
  - `interfaces/` – Defines shared model interfaces.  
  - `weights/` – Stores trained model weights.  
- **requirements.txt** – Lists dependencies needed for this project.

📌 **For a detailed explanation, refer to** `MNIST_Classifiers/README.md`.

---

## **2. NLP_Vision_AnimalRecognition**
This project integrates **Named Entity Recognition (NER) and Image Classification** to verify whether a given text description correctly identifies an animal in an image.

### **Main Components:**
- **datasets/** – Contains image and text datasets used for training.
- **models/** – Includes implementations of both the NER and image classification models.
- **notebooks/** – Jupyter notebooks for training, testing, and demo visualization.
- **src/**  
  - `inferences/` – Code for model inference.  
  - `weights/` – Stores trained model weights.  
- **pipeline.py** – The main script that runs the multimodal verification pipeline.
- **requirements.txt** – Lists dependencies required for this project.

📌 **For a detailed explanation, refer to** `NLP_Vision_AnimalRecognition/README.md`.

---

## **Installation & Setup**
Each project has its own dependencies, so before running any code, make sure to install the necessary packages for the specific project:

```bash
cd MNIST_Classifiers
pip install -r requirements.txt

cd NLP_Vision_AnimalRecognition
pip install -r requirements.txt
```