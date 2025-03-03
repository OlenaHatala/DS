import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        """
        Initializes a Node in the decision tree.

        Args:
            feature (int): Index of the feature for splitting at this node.
            threshold (float): Threshold value for splitting the feature.
            left (Node): Left child node.
            right (Node): Right child node.
            value (int, float): The value (class label) if this node is a leaf.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Checks if the node is a leaf node (i.e., it holds a class label).

        Returns:
            bool: True if the node is a leaf, otherwise False.
        """
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        Initializes the DecisionTree classifier.

        Args:
            min_samples_split (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the tree.
            n_features (int): Number of features to consider when looking for the best split.
        """
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None

    def fit(self, X, y):
        """
        Fits the decision tree model to the training data.

        Args:
            X (array-like): Training feature data.
            y (array-like): Training labels.
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively builds the tree.

        Args:
            X (array-like): Features of the dataset.
            y (array-like): Labels of the dataset.
            depth (int): The current depth of the tree.

        Returns:
            Node: The root node of the decision tree.
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    def _best_split(self, X, y, feat_idxs):
        """
        Finds the best feature and threshold to split the data.

        Args:
            X (array-like): Feature data.
            y (array-like): Labels.
            feat_idxs (array): List of feature indices to consider.

        Returns:
            tuple: The best feature index and the threshold to split on.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold


    def _information_gain(self, y, X_column, threshold):
        """
        Calculates the information gain for a particular split.

        Args:
            y (array-like): Labels of the dataset.
            X_column (array-like): A specific column (feature) of the dataset.
            threshold (float): Threshold value for splitting the feature.

        Returns:
            float: The information gain after the split.
        """
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        """
        Splits the data based on the threshold value.

        Args:
            X_column (array-like): A feature column.
            split_thresh (float): The threshold for splitting the data.

        Returns:
            tuple: Indices of the left and right splits.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """
        Calculates the entropy of the labels.

        Args:
            y (array-like): The labels.

        Returns:
            float: The entropy of the labels.
        """
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])


    def _most_common_label(self, y):
        """
        Finds the most common label in the dataset.

        Args:
            y (array-like): The labels.

        Returns:
            int: The most common label.
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Args:
            X (array-like): The input data for prediction.

        Returns:
            array: The predicted class labels.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverses the tree recursively to predict the class for a given input.

        Args:
            x (array-like): The input data.
            node (Node): The current node in the tree.

        Returns:
            int: The predicted class label.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


