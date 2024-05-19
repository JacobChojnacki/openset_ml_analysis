import time


def fit_model(model, X_train):
    """
    Fit the given model on the training data.

    Parameters:
    - model: The model object to fit
    - X_train: Training data
    - nproj: Number of projections for the model (default=100)
    """
    start_time = time.time()
    model.fit(X_train)
    print('Model fitting completed in %.2f seconds.' % (time.time() - start_time))


def score_data(model, data):
    """
    Calculate scores for the given data using the trained model.

    Parameters:
    - model: The trained model object
    - data: Data to be scored

    Returns:
    - scores: Scores for the data
    """
    try:
        scores = [model.score(row) for row in data]
    except Exception:
        scores = model.score_samples(data)
    return scores


def fit_and_score_model(model, X_train, X_test, y_train):
    """
    Fit the given model on the training data and calculate scores for train, test, and out-of-distribution (ood) data.

    Parameters:
    - model: The model object to fit and score
    - X_train: Training data
    - X_test: Test data
    - y_train: Target labels for training data
    - nproj: Number of projections for the model (default=100)

    Returns:
    - train_scores: Scores for the training data
    - test_scores: Scores for the test data
    - ood_scores: Scores for the out-of-distribution data
    """
    fit_model(model, X_train)

    print('Scoring train data...')
    train_scores = score_data(model, X_train)
    print('Scoring test data...')
    test_scores = score_data(model, X_test)
    print('Scoring out-of-distribution data...')
    ood_scores = score_data(model, X_train[y_train.T[0] == 1])

    return train_scores, test_scores, ood_scores
