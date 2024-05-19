import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from openset.tools import dataset_tool
from sklearn.metrics import RocCurveDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, normalize=True, cmap='coolwarm', figsize=(10, 8), annot_kws=None, ax=None):
    if annot_kws is None:
        annot_kws = {'size': 10, 'color': 'black'}
    cm = confusion_matrix(y_true, y_pred, normalize='all' if normalize else None)
    n_classes = len(cm)
    palette = sns.color_palette(cmap, as_cmap=True)
    mask = np.eye(n_classes, dtype=bool)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='.02%' if normalize else 'd',
        cbar_kws={'label': 'Percentage' if normalize else None},
        linewidths=1,
        linecolor='gray',
        square=True,
        annot_kws=annot_kws,
        cmap=palette,
        mask=mask,
        ax=ax,
    )
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)


def plot_roc_curve(
    X, y, y_pred=None, pos_label=0, title='Receiver Operating Characteristic', color='darkorange', ax=None
):
    """
    Plot ROC curve for a given model.

    Parameters:
    - model: The trained model object with a `fit()` method.
    - X: Input features for the model.
    - y: True labels.
    - pos_label: The label to consider as positive.
    - title: Title for the plot.
    - color: Color for the ROC curve.
    """
    sns.set_style('whitegrid')
    RocCurveDisplay.from_predictions(
        y_true=y, y_pred=y_pred, color=color, name='LOF', linestyle='-', linewidth=2, ax=ax
    )
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title='Model', fontsize=10, title_fontsize=10, loc='lower right')


def visualize_3d_data(df_subset, x, y, z, label, label_values, ax):
    df_misclassified = df_subset[df_subset['y'] != df_subset['y_pred']]

    colors = sns.color_palette('husl', 6)
    for idx, val in enumerate(label_values):
        subset = df_subset[df_subset[label] == val]
        ax.scatter(subset[x], subset[y], subset[z], label=str(val), color=colors[idx + 2], s=50, alpha=0.8)

    ax.scatter(
        df_misclassified[x],
        df_misclassified[y],
        df_misclassified[z],
        label='Misclassified',
        color='red',
        marker='x',
        s=100,
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend()

    ax.grid(True)  # Add grid for better perspective
    ax.view_init(elev=20, azim=45)  # Set view angle

    ax.set_title('3D Visualization of Data with Labels')
    ax.legend()


def plot_boxplot(train_scores, test_scores, ood_scores, show_means=False, show_outliers=False, show_notches=False):
    """
    Plot a boxplot to visualize the distribution of scores across different datasets.

    Parameters:
        train_scores (list): List of scores for the training dataset.
        test_scores (list): List of scores for the test dataset.
        ood_scores (list): List of scores for the out-of-distribution dataset.
        show_means (bool): Whether to show the mean line on the boxplot.
        show_outliers (bool): Whether to show outliers in the boxplot.
        show_notches (bool): Whether to show notches indicating confidence intervals around the median.
    """
    # Define data and labels
    labels = ['Train', 'Test', 'Out of Distribution']
    data = [train_scores, test_scores, ood_scores]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Customize boxplot appearance
    boxprops = {'color': 'blue', 'linewidth': 2}
    whiskerprops = {'color': 'black', 'linewidth': 1.5}
    medianprops = {'color': 'red', 'linewidth': 2}
    capprops = {'color': 'black', 'linewidth': 1.5}

    # Plot boxplot
    ax.boxplot(
        data,
        labels=labels,
        showmeans=show_means,
        showfliers=show_outliers,
        notch=show_notches,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
        capprops=capprops,
    )

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set labels and title
    ax.set_ylabel('Scores')
    ax.set_xlabel('Dataset')
    ax.set_title('Distribution of Scores across Datasets')

    # Show plot
    plt.show()


def visualize_results_3D_plane(X_train, y_train, X_test, y_test, model, y_pred_train, lof=True, irw=False):
    """
    Visualizes the results  on train and test data using t-SNE for dimensionality reduction.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    X_test (array-like): Test data features.
    y_test (array-like): Test data labels.
    lof_model_g (object): Fitted Local Outlier Factor model.
    lof_y_pred_train_g (array-like): Predicted labels for training data.
    lof_y_pred_test_g (array-like): Predicted labels for test data.

    Returns:
    None
    """
    subset_train = dataset_tool.perform_tsne(
        X_train, y_train, y_pred_train, n_components=3, random_state=42, verbose=False
    )
    if lof:
        model.novelty = True
        y_pred_test = model._predict(X_test)
    else:
        y_pred_test = model.predict(X_test)

    if not irw:
        y_pred_test[y_pred_test == 1] = 0  # inliers
        y_pred_test[y_pred_test == -1] = 1  # outliers

    subset_test = dataset_tool.perform_tsne(
        X_test, y_test, y_pred_test, n_components=3, random_state=42, verbose=False, perplexity=len(X_test) - 1
    )

    _, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(15, 10))
    visualize_3d_data(subset_train, 'tsne-1-d', 'tsne-2-d', 'tsne-3-d', 'y', subset_train['y'].unique(), ax=axs[0])
    visualize_3d_data(subset_test, 'tsne-1-d', 'tsne-2-d', 'tsne-3-d', 'y', subset_test['y'].unique(), ax=axs[1])


def plot_outlier_detection_results(model, X_train, y_train, X_test, y_test, train_pred, title='LOF'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Title for the entire plot
    fig.suptitle(f'Outlier Detection Using {title}', fontsize=16)

    # Plot training data
    plot_confusion_matrix(y_train, train_pred, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix - Training Data')

    plot_roc_curve(X_train, y_train, train_pred, ax=axes[0, 1])
    axes[0, 1].set_title('ROC Curve - Training Data')

    if title.lower() == 'lof':
        model.novelty = True
        test_pred = model._predict(X_test)
    else:
        test_pred = model.predict(X_test)

    if title.lower() != 'irw':
        test_pred[test_pred == 1] = 0  # inliers
        test_pred[test_pred == -1] = 1  # outliers

    # Plot testing data
    plot_confusion_matrix(y_test, test_pred, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix - Testing Data')

    plot_roc_curve(X_test, y_test, test_pred, ax=axes[1, 1])
    axes[1, 1].set_title('ROC Curve - Testing Data')

    plt.tight_layout()
    plt.show()


def lof_visualize_bar(df_train: pd.DataFrame, df_test: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(18, 12))

    palette = sns.color_palette(['#4C72B0', '#DD8452'])
    colors = sns.color_palette(palette, n_colors=2)

    _ = sns.barplot(
        x=df_train.index, y=df_train['y_score'], hue=df_train['is_incorrect'], palette=colors, dodge=False, ax=ax[0]
    )

    _ = sns.barplot(
        x=df_test.index, y=df_test['y_score'], hue=df_test['is_incorrect'], palette=colors, dodge=False, ax=ax[1]
    )

    plt.xlabel('Index', fontsize=14)
    plt.ylabel('Prediction Score', fontsize=14)
    plt.title('Prediction Scores with Incorrect Predictions Highlighted', fontsize=16)

    plt.xticks(rotation=90)

    plt.legend(title='Prediction', title_fontsize='14', fontsize='12', loc='upper right')

    plt.grid(axis='y', linestyle='--', alpha=0.5)

    sns.despine()

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
