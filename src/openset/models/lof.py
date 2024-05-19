from typing import Any

import numpy as np
import pandas as pd

from openset.tools import visualization_tool
from sklearn.neighbors import LocalOutlierFactor


def detect_outliers(
    X_train,
    novelty: bool = False,
    n_neighbors: int = 15,
    contamination: str = 'auto',
    leaf_size: Any = 30,
    metric: Any = 'minkowski',
    p: Any = 2,
):
    """
    Detect outliers in a dataset using Local Outlier Factor algorithm.

    Parameters:
    X_train : array-like, shape (n_samples, n_features)
        The training input samples.

    n_neighbors : int, optional (default=15)
        Number of neighbors to consider for each sample.

    contamination : float or 'auto', optional (default='auto')
        Proportion of outliers in the dataset, or 'auto' to estimate from the data.

    Returns:
    y_pred : array-like, shape (n_samples,)
        Predicted labels (0: inlier, 1: outlier).

    pred_scores : array-like, shape (n_samples,)
        Outlier scores.
    """
    # Initialize and fit the model
    model = LocalOutlierFactor(
        novelty=novelty, n_neighbors=n_neighbors, contamination=contamination, leaf_size=leaf_size, metric=metric, p=p
    )
    y_pred = model.fit_predict(X_train)

    # Convert outlier labels
    y_pred[y_pred == 1] = 0  # inliers
    y_pred[y_pred == -1] = 1  # outliers

    # Get outlier scores
    pred_scores = -1 * model.negative_outlier_factor_

    return y_pred, pred_scores, model


def visualize_outliers_lof(
    X_train,
    X_test,
    y_train,
    y_test,
    train_pred=None,
    test_pred=None,
    novelty=False,
    n_neighbors=17,
    contamination='auto',
    leaf_size=25,
    metric='minkowski',
    p=1,
):
    """
    Visualizes Outliers Detected by LOF (Local Outlier Factor) Algorithm

    Parameters:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.
        train_pred (array-like, optional): Predictions on training data. If None, predictions will be made.
        test_pred (array-like, optional): Predictions on testing data. If None, predictions will be made.
        novelty (bool, optional): If True, assumes test data is novel. Default is False.
        n_neighbors (int, optional): Number of neighbors for LOF algorithm. Default is 17.
        contamination (float or 'auto', optional): Proportion of outliers in the data. Default is 'auto'.
        leaf_size (int, optional): Leaf size for the tree structure. Default is 25.
        metric (str, optional): Distance metric to use. Default is "minkowski".
        p (int, optional): Parameter for the Minkowski metric. Default is 1.
    """
    # If predictions are not provided, make predictions
    if train_pred is None or test_pred is None:
        train_pred, _, model = detect_outliers(
            X_train,
            novelty=novelty,
            n_neighbors=n_neighbors,
            contamination=contamination,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
        )
    # Plot training data
    visualization_tool.plot_outlier_detection_results(model, X_train, y_train, X_test, y_test, train_pred=train_pred)

    return 0


def lof_dataframe(y_true: np.array, y_pred: np.array, y_score: np.array):
    """Create dataframe with results of LOF algorithm

    Args:
        y_true (np.array): Groud truth labels
        y_pred (np.array): Predicted labels
        y_score (np.array): Scored values
    """
    df = pd.DataFrame([y_true.reshape(len(y_true)), y_pred, y_score]).T
    df.columns = ['y_true', 'y_pred', 'y_score']
    df['is_incorrect'] = df['y_pred'] != df['y_true']
    return df


def lof_predict_test(model, X_data: np.array):
    """Make prediction for test data

    Args:
        model (LocalOutlierFactor): model trained on train data
        X_data (np.array): train data
    Returns:
        y_pred (np.array): predicted data
        y_score (np.array): scored data
    """
    model.novelty = True
    y_pred = model._predict(X_data)
    y_pred[y_pred == 1] = 0  # inliers
    y_pred[y_pred == -1] = 1  # outliers
    y_score = -1 * model.score_samples(X_data)
    return y_pred, y_score
