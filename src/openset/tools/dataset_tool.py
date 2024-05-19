from pathlib import Path

import pandas as pd
import scipy.io

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


def load_data(data_path='../data/wine.mat', test_size=0.2, random_state=42):
    dataset_path = Path(data_path)
    dataset = scipy.io.loadmat(dataset_path)

    # Extract features (X) and labels (y)
    X = dataset['X']
    y = dataset['y']

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def perform_tsne(X_train, y_train, y_pred, n_components=3, random_state=0, verbose=True, perplexity=30):
    """
    Perform t-SNE transformation on the input data and return a DataFrame with the transformed features.

    Parameters:
    - X_train: Input data for t-SNE transformation
    - y_train: Target labels
    - n_components: Number of components for t-SNE transformation (default=3)
    - random_state: Random state for reproducibility (default=0)
    - verbose: Verbosity mode (default=True)

    Returns:
    - df_subset: DataFrame containing the transformed features
    """
    # Initialize t-SNE with specified parameters
    tsne = TSNE(n_components=n_components, random_state=random_state, verbose=verbose, perplexity=perplexity)

    # Fit and transform the input data
    tsne_results = tsne.fit_transform(X_train)

    # Create a DataFrame to store the transformed features
    df_subset = pd.DataFrame(tsne_results, columns=[f'tsne-{i + 1}-d' for i in range(n_components)])

    # Add the target labels to the DataFrame
    df_subset['y'] = y_train
    df_subset['y_pred'] = y_pred
    return df_subset
