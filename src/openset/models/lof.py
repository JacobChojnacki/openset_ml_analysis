from typing import Any

import matplotlib.pyplot as plt

from openset.tools import visualization_tool
from sklearn.neighbors import LocalOutlierFactor


def detect_outliers(X_train,
                    novelty: bool = False,
                    n_neighbors: int = 15,
                    contamination: str = 'auto',
                    leaf_size: Any = 30,
                    metric: Any = "minkowski",
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
    model = LocalOutlierFactor(novelty=novelty,
                               n_neighbors=n_neighbors,
                               contamination=contamination,
                               leaf_size=leaf_size,
                               metric=metric,
                               p=p)
    y_pred = model.fit_predict(X_train)

    # Convert outlier labels
    y_pred[y_pred == 1] = 0  # inliers
    y_pred[y_pred == -1] = 1  # outliers

    # Get outlier scores
    pred_scores = -1 * model.negative_outlier_factor_

    return y_pred, pred_scores, model


def visualize_outliers_lof(X_train, X_test, y_train, y_test, train_pred=None, test_pred=None, novelty=False,
                           n_neighbors=17, contamination='auto', leaf_size=25, metric="minkowski", p=1):
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
        train_pred, train_score, model = detect_outliers(X_train, novelty=novelty, n_neighbors=n_neighbors,
                                                               contamination=contamination, leaf_size=leaf_size,
                                                               metric=metric, p=p)

    # Plot confusion matrix and ROC curve
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Title for the entire plot
    fig.suptitle('Outlier Detection Using LOF', fontsize=16)

    # Plot training data
    visualization_tool.plot_confusion_matrix(y_train, train_pred, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix - Training Data')

    visualization_tool.plot_roc_curve(X_train, y_train, train_pred, ax=axes[0, 1])
    axes[0, 1].set_title('ROC Curve - Training Data')

    model.novelty = True
    test_pred = model._predict(X_test)
    test_pred[test_pred == 1] = 0  # inliers
    test_pred[test_pred == -1] = 1  # outliers

    # Plot testing data
    visualization_tool.plot_confusion_matrix(y_test, test_pred, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix - Testing Data')

    visualization_tool.plot_roc_curve(X_test, y_test, test_pred, ax=axes[1, 1])
    axes[1, 1].set_title('ROC Curve - Testing Data')

    plt.tight_layout()
    plt.show()


