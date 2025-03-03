import numpy as np
import joblib
from collections import Counter
from classifiers.__decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None, **kwargs):
        """
        Initializes the RandomForest model with the given hyperparameters.

        Args:
            n_trees (int): The number of decision trees in the forest.
            max_depth (int): The maximum depth of each decision tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            n_feature (int): The number of features to consider when looking for the best split for each tree.
            **kwargs: Additional parameters that can be passed to the constructor.
        """        
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.n_features=n_feature
        self.trees = []

    def train(self, x_train, y_train, **kwargs):
        """
        Trains the RandomForest model by fitting multiple decision trees on bootstrap samples of the data.

        Args:
            x_train (array-like): The training features.
            y_train (array-like): The training labels.
            **kwargs: Additional parameters for the `fit` method of DecisionTree.
        """        
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                            n_features=self.n_features)
            X_sample, y_sample = self._bootstrap_samples(x_train, y_train)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, x_test):
        """
        Makes predictions using the RandomForest model by averaging predictions from all decision trees.

        Args:
            x_test (array-like): The test features.

        Returns:
            array: The predicted class labels from the random forest (majority vote from all trees).
        """        
        predictions = np.array([tree.predict(x_test) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions

    def _bootstrap_samples(self, X, y):
        """
        Generates bootstrap samples from the original dataset (sampling with replacement).

        Args:
            X (array-like): The feature data.
            y (array-like): The label data.

        Returns:
            tuple: The bootstrap sample of features (X_sample) and labels (y_sample).
        """
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def _most_common_label(self, y):
        """
        Finds the most common label in the predicted labels (used for majority voting).

        Args:
            y (array-like): The predicted labels.

        Returns:
            int: The most common label (class).
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def save(self, filepath):
        """
        Saves the trained RandomForest model to a file using joblib.

        Args:
            filepath (str): The path where the model should be saved.
        """
        joblib.dump(self, filepath)
        print(f"RandomForest model saved to {filepath}")

    
    def load(self, filepath):
        """
        Loads a previously saved RandomForest model from a file using joblib and updates the current model.

        Args:
            filepath (str): The path from which the model should be loaded.
        """
        loaded_model = joblib.load(filepath)
        
        self.__dict__.update(loaded_model.__dict__)
        print(f"RandomForest model loaded and updated from {filepath}")