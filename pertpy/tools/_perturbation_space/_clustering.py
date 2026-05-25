from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import pairwise_distances

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace, _resolve_matrix

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData


class ClusteringSpace(PerturbationSpace):
    """Applies various clustering techniques to an embedding."""

    def evaluate_clustering(
        self,
        adata: AnnData,
        true_label_col: str,
        cluster_col: str,
        metrics: Iterable[str] = None,
        *,
        layer_key: str | None = None,
        embedding_key: str | None = None,
        **kwargs,
    ):
        """Evaluation of previously computed clustering against ground truth labels.

        Args:
            adata: AnnData object that contains the clustered data and the cluster labels.
            true_label_col: ground truth labels.
            cluster_col: cluster computed labels.
            metrics: Metrics to compute. If `None` it defaults to ``["nmi", "ari", "asw"]`` — the canonical
                trio for clustering benchmarks (mutual information, agreement, silhouette).
            layer_key: Layer to resolve cell coordinates from when computing ASW.
            embedding_key: Embedding to resolve cell coordinates from when computing ASW.
            **kwargs: Additional arguments to pass to the metrics. For nmi, average_method can be passed.
                For asw, ``metric``, ``distances``, ``sample_size``, and ``random_state`` can be passed.

        Examples:
            Example usage with KMeansSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> kmeans = pt.tl.KMeansSpace()
            >>> kmeans_adata = kmeans.compute(mdata["rna"], n_clusters=26)
            >>> results = kmeans.evaluate_clustering(
            ...     kmeans_adata, true_label_col="gene_target", cluster_col="k-means", metrics=["nmi"]
            ... )
        """
        if metrics is None:
            metrics = ["nmi", "ari", "asw"]
        true_labels = adata.obs[true_label_col]

        results: dict[str, float] = {}
        for metric in metrics:
            if metric == "nmi":
                from pertpy.tools._perturbation_space._metrics import nmi

                if "average_method" not in kwargs:
                    kwargs["average_method"] = "arithmetic"  # by default in sklearn implementation

                results["nmi"] = nmi(
                    true_labels=true_labels,
                    predicted_labels=adata.obs[cluster_col],
                    average_method=kwargs["average_method"],
                )

            elif metric == "ari":
                from pertpy.tools._perturbation_space._metrics import ari

                results["ari"] = ari(true_labels=true_labels, predicted_labels=adata.obs[cluster_col])

            elif metric == "asw":
                from pertpy.tools._perturbation_space._metrics import asw

                kwargs.setdefault("metric", "euclidean")
                kwargs.setdefault("sample_size", None)
                kwargs.setdefault("random_state", None)

                if "distances" in kwargs:
                    distances = kwargs["distances"]
                else:
                    distances = pairwise_distances(
                        _resolve_matrix(adata, layer_key=layer_key, embedding_key=embedding_key),
                        metric=kwargs["metric"],
                    )

                results["asw"] = asw(
                    pairwise_distances=distances,
                    labels=true_labels,
                    metric=kwargs["metric"],
                    sample_size=kwargs["sample_size"],
                    random_state=kwargs["random_state"],
                )

        return results
