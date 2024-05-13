#!/usr/bin/env python
# coding: utf-8

import numpy as np

from openset.tools import visualization_tool


class IRWDepth:
    """Integrated Rank-Weighted depth based on NIPS2022 paper P.Colombo et al.,
    Beyond Mahalanobis-Based Scores for Textual OOD Detection (Monte Carlo approximation)
    """

    def __init__(self, contamination=0.1):
        """
        Initialize IRWDepth model.

        Parameters:
        - contamination (float): The proportion of outliers in the dataset.
        """
        self.contamination = contamination
        self._threshold = 0.1

    def fit(self, X: np.ndarray, nproj=1000):
        """
        Fit IRWDepth model to the training data.

        Parameters:
        - X (np.ndarray): Training samples with shape (n_samples, n_features).
        - nproj (int): Number of random vectors of hypersphere S.

        Returns:
        - bool: True if the model is fitted successfully.
        """
        self.n_samples, self.n_features = X.shape
        self.nproj = nproj

        self.U = np.random.normal(size=(self.n_features, self.nproj))
        self.U /= np.linalg.norm(self.U, axis=0)  # Normalize random vectors

        self.M = X @ self.U  # Compute M = XU

        print(f'Fitted IRWDepth model with {self.nproj} projections in {self.n_features} dimensions.')

        return True

    def score(self, x: np.ndarray) -> float:
        """
        Calculate IRWDepth score for a single sample.

        Parameters:
        - x (np.ndarray): Input sample with shape (n_features,).

        Returns:
        - float: IRWDepth score for the sample.
        """
        v = x @ self.U
        M_v = self.M - v
        counts = np.minimum((M_v <= 0).sum(axis=0), (M_v > 0).sum(axis=0))
        return counts.mean() / self.n_samples

    def predict(self, X: np.ndarray, return_irw_scores=False) -> np.ndarray:
        """
        Predict labels (inliers or outliers) of input samples.

        Parameters:
        - X (np.ndarray): Input samples with shape (n_samples, n_features).
        - return_irw_scores (bool): Whether to return IRWDepth scores along with labels.

        Returns:
        - np.ndarray: Predicted labels for input samples.
        """
        irw_scores = np.array([self.score(x) for x in X])

        # Determine the threshold dynamically based on the actual proportion of outliers

        # Classify samples
        labels = np.where(irw_scores >= self._threshold, 0, 1)
        if return_irw_scores:
            return labels, irw_scores
        else:
            return labels


def visualize_outliers_irw(X_train, X_test, y_train, y_test, train_pred=None, model=None):
    """
    Visualizes Outliers Detected by LOF (Local Outlier Factor) Algorithm

    Parameters:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.
        train_pred (array-like, optional): Predictions on training data. If None, predictions will be made.
        test_pred (array-like, optional): Predictions on testing data. If None, predictions will be made.
        contamination (float, optional): Proportion of outliers in the data. Default is 'auto'.
        random_state (int, optional): Seed for random number generator. Defaults to 42.
    """
    # If predictions are not provided, make predictions
    train_pred = model.predict(X_train) if train_pred is None else train_pred
    visualization_tool.plot_outlier_detection_results(model, X_train, y_train, X_test, y_test, train_pred=train_pred,
                                                      title="IRW")

    return 0
