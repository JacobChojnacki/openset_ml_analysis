from openset.tools import visualization_tool
from sklearn.covariance import EllipticEnvelope


def detect_outliers(X_train, contamination: float = 0.1, random_state: int = 42):
    """
    Detect outliers in a dataset using Eliptic Envelope algorithm.

    Parameters:
    X_train : array-like, shape (n_samples, n_features)
        The training input samples.


    contamination : float or 'auto', optional (default='auto')
        Proportion of outliers in the dataset, or 'auto' to estimate from the data.

    Returns:
    y_pred : array-like, shape (n_samples,)
        Predicted labels (0: inlier, 1: outlier).

    pred_scores : array-like, shape (n_samples,)
        Outlier scores.
    """
    # Initialize and fit the model
    model = EllipticEnvelope(contamination=contamination, random_state=random_state)

    y_pred = model.fit_predict(X_train)

    # Convert outlier labels
    y_pred[y_pred == 1] = 0  # inliers
    y_pred[y_pred == -1] = 1  # outliers

    # Get outlier scores

    return y_pred, model


def visualize_outliers_mahalanobis(
    X_train, X_test, y_train, y_test, train_pred=None, test_pred=None, contamination=0.1, random_state: int = 42
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
        contamination (float, optional): Proportion of outliers in the data. Default is 'auto'.
        random_state (int, optional): Seed for random number generator. Defaults to 42.
    """
    # If predictions are not provided, make predictions
    if train_pred is None or test_pred is None:
        train_pred, model = detect_outliers(X_train, contamination=contamination, random_state=random_state)
        # Plot training data
    visualization_tool.plot_outlier_detection_results(
        model, X_train, y_train, X_test, y_test, train_pred=train_pred, title='Mahalanobis'
    )

    return 0
