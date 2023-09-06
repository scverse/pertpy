from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

if TYPE_CHECKING:
    import numpy as np
    from numpy._typing import ArrayLike


def nmi(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    average_method: Literal["min", "max", "geometric", "arithmetic"] = "arithmetic",
) -> float:
    """Calculates the normalized mutual information score between two sets of clusters.

    See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html

    Args:
        true_labels: A clustering of the data into disjoint subsets.
        predicted_labels: A clustering of the data into disjoint subsets.
        average_method: How to compute the normalizer in the denominator.

    Returns:
        Score between 0.0 and 1.0 in normalized nats (based on the natural logarithm). 1.0 stands for perfectly complete labeling.
    """
    return normalized_mutual_info_score(
        labels_true=true_labels, labels_pred=predicted_labels, average_method=average_method
    )


def ari(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Calculates the adjusted rand index for chance.

    See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    Args:
        true_labels: Ground truth class labels to be used as a reference.
        predicted_labels: Cluster labels to evaluate.

    Returns:
        Similarity score between -0.5 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.
    """
    return adjusted_rand_score(labels_true=true_labels, labels_pred=predicted_labels)


def asw(
    pairwise_distances: ArrayLike,
    labels: ArrayLike,
    metric: str = "euclidean",
    sample_size: int = None,
    random_state: int = None,
    **kwargs,
) -> float:
    """Computes the average-width silhouette score.

    See: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    Args:
        pairwise_distances: An array of pairwise distances between samples, or a feature array.
        labels: Predicted labels for each sample.
        metric: The metric to use when calculating distance between instances in a feature array.
                If metric is a string, it must be one of the options allowed by metrics.pairwise.pairwise_distances.
                If X is the distance array itself, use metric="precomputed".
        sample_size: The size of the sample to use when computing the Silhouette Coefficient on a random subset of the data.
                     If sample_size is None, no sampling is used.
        random_state: Determines random number generation for selecting a subset of samples. Used when sample_size is not None.
        **kwargs: Any further parameters are passed directly to the distance function. If using a scipy.spatial.distance metric, the parameters are still metric dependent.

    Returns:
        Mean Silhouette Coefficient for all samples.
    """
    return silhouette_score(
        X=pairwise_distances, labels=labels, metric=metric, sample_size=sample_size, random_state=random_state, **kwargs
    )
