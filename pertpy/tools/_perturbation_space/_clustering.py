from typing import Literal


class ClusteringSpace:
    """Applies various clustering techniques to an embedding."""

    def __call__(
        self,
        *args,
        clustering_fct: Literal["dendrogram", "k-means", "dbscan", "spectral clustering", "gaussian mixture"],
        **kwargs
    ):
        # TODO implement
        pass
