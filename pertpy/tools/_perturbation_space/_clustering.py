from typing import List

from anndata import AnnData
from sklearn.metrics import pairwise_distances

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


class ClusteringSpace(PerturbationSpace):
    """Applies various clustering techniques to an embedding."""

    def __init__(self):
        self.X = None

    def evaluate_clustering(
        self,
        adata: AnnData,
        true_label_col: str,
        cluster_col: str,
        metrics: List[str] = None,
        **kwargs,
    ):
        """Evaluation of previously computed clustering against ground truth labels.

        Args:
        ----
            adata: AnnData object that contains the clustered data and the cluster labels.
            true_label_col: ground truth labels.
            cluster_col: cluster computed labels.
            metrics: Metrics to compute. Defaults to ['nmi', 'ari', 'asw'].
        """
        if metrics is None:
            metrics = ["nmi", "ari", "asw"]
        true_labels = adata.obs[true_label_col]

        results = {}
        for metric in metrics:
            if metric == "nmi":
                from pertpy.tools._perturbation_space._metrics import nmi

                if "average_method" not in kwargs:
                    kwargs["average_method"] = "arithmetic"  # by default in sklearn implementation

                nmi_score = nmi(
                    true_labels=true_labels,
                    predicted_labels=adata.obs[cluster_col],
                    average_method=kwargs["average_method"],
                )
                results["nmi"] = nmi_score

            if metric == "ari":
                from pertpy.tools._perturbation_space._metrics import ari

                ari_score = ari(true_labels=true_labels, predicted_labels=adata.obs[cluster_col])
                results["ari"] = ari_score

            if metric == "asw":
                from pertpy.tools._perturbation_space._metrics import asw

                if "metric" not in kwargs.keys():
                    kwargs["metric"] = "euclidean"
                if "distances" not in kwargs.keys():
                    distances = pairwise_distances(self.X, metric=kwargs["metric"])
                if "sample_size" not in kwargs.keys():
                    kwargs["sample_size"] = None
                if "random_state" not in kwargs.keys():
                    kwargs["random_state"] = None

                asw_score = asw(
                    pairwise_distances=distances,
                    labels=true_labels,
                    metric=kwargs["metric"],
                    sample_size=kwargs["sample_size"],
                    random_state=kwargs["random_state"],
                )

                results["asw"] = asw_score

        return results
