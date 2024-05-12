import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_curve


def plot_confusion_matrix(y_true, y_pred, normalize=True, cmap="coolwarm", figsize=(10, 8),
                          annot_kws={"size": 10, "color": 'black'}, ax=None):
    cm = confusion_matrix(y_true, y_pred, normalize='all' if normalize else None)
    n_classes = len(cm)
    palette = sns.color_palette(cmap, as_cmap=True)
    mask = np.eye(n_classes, dtype=bool)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".02%" if normalize else 'd',
                cbar_kws={'label': 'Percentage' if normalize else None},
                linewidths=1, linecolor='gray', square=True, annot_kws=annot_kws,
                cmap=palette, mask=mask, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=16)
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)


def plot_roc_curve(X, y, y_pred=None, pos_label=0, title="Receiver Operating Characteristic", color='darkorange', ax=None):
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
    sns.set_style("whitegrid")
    roc_display = RocCurveDisplay.from_predictions(y_true=y, y_pred=y_pred, color=color, name="LOF", linestyle='-', linewidth=2, ax=ax)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(title="Model", fontsize=10, title_fontsize=10, loc='lower right')


def visualize_data_3d(df_subset, x, y, z, label, label_values):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    df_misclassified = df_subset[df_subset['y'] != df_subset['y_pred']]

    colors = sns.color_palette("husl", 6)
    for idx, val in enumerate(label_values):
        subset = df_subset[df_subset[label] == val]
        ax.scatter(subset[x], subset[y], subset[z], label=str(val), color=colors[idx+2], s=50, alpha=0.8)

    ax.scatter(df_misclassified[x], df_misclassified[y], df_misclassified[z],
               label='Misclassified', color='red', marker='x', s=100)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.legend()

    ax.grid(True)  # Add grid for better perspective
    ax.view_init(elev=20, azim=45)  # Set view angle

    plt.title("3D Visualization of Data with Labels")
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_boxplot(train_scores,
                 test_scores,
                 ood_scores,
                 show_means=False,
                 show_outliers=False,
                 show_notches=False):
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
    boxprops = {"color": 'blue', "linewidth": 2}
    whiskerprops = {"color": 'black', "linewidth": 1.5}
    medianprops = {"color": 'red', "linewidth": 2}
    capprops = {"color": 'black', "linewidth": 1.5}

    # Plot boxplot
    ax.boxplot(data, labels=labels, showmeans=show_means, showfliers=show_outliers,
               notch=show_notches, boxprops=boxprops, whiskerprops=whiskerprops,
               medianprops=medianprops, capprops=capprops)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set labels and title
    ax.set_ylabel('Scores')
    ax.set_xlabel('Dataset')
    ax.set_title('Distribution of Scores across Datasets')

    # Show plot
    plt.show()
