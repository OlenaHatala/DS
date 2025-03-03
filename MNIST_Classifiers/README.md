# MNIST Classifiers
---


## Requirements

To run the project, you need to install the dependencies. The project supports both `pip` and `conda` package managers. 

### Option 1: Install dependencies with `pip`

1. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv env
    ```
2. Activate the virtual environment:
    - On Windows:
      ```bash
      .\env\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source env/bin/activate
      ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Install dependencies with `conda`

1. Create a new conda environment:
    ```bash
    conda create --name testtasks python=3.8
    ```
2. Activate the conda environment:
    ```bash
    conda activate testtasks
    ```
3. Install the required dependencies from `requirements.txt`:
    ```bash
    conda install --file requirements.txt
    ```


## Running the Models

After installing the dependencies, you can run the models in the `demo.ipynb` Jupyter notebook. The notebook contains step-by-step instructions for loading the MNIST dataset, training the models, and evaluating their performance.


### Additional Notes

- The project includes Jupyter notebooks located in the `notebooks/` directory.

- The models' implementations are located in the `src/classifiers/` directory. This includes:
  - `cnn.py`: Convolutional Neural Network (CNN) model.
  - `ffnn.py`: Feed-Forward Neural Network (FFNN) model.
  - `random_forest.py`: Random Forest model.
  - `mnist_classifier.py`: A script that uses the aforementioned models to classify MNIST digits.

- The `src/interfaces/` directory contains the `mnist_classifier_interface.py` file, which defines the interface for the classifier models, allowing you to switch between different classifiers with ease.

- The models include pre-trained weights, which can be loaded directly from the `src/weights/` directory.

