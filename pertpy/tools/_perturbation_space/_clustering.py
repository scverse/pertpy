from typing import List, Literal
from abc import ABC, abstractmethod

from anndata import AnnData
from sklearn.metrics import pairwise_distances

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


class ClusteringSpace(ABC):
    """Applies various clustering techniques to an embedding."""
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(
        self,
        adata: AnnData,
        true_label_col: str,
        cluster_col: str,
        metrics: List[str] = None,
        **kwargs,
    ):
        """Evaluation of previously computed clustering against ground truth labels

        Args:
            adata: adata that contains the clustered data and the cluster labels
            true_label_col: ground truth labels
            cluster_col: cluster computed labels
            metrics: Defaults to ['nmi', 'ari', 'asw'].
        """
        if metrics is None:
            metrics = ["nmi", "ari", "asw"]
        true_labels = adata.obs[true_label_col]

        results = {}
        for metric in metrics:
            if metric == "nmi":
                from pertpy.tools._perturbation_space._metrics import nmi

                nmi_score = nmi(true_labels=true_labels, predicted_labels=adata.obs[cluster_col], **kwargs)
                results["nmi"] = nmi_score

            if metric == "ari":
                from pertpy.tools._perturbation_space._metrics import ari

                ari_score = ari(true_labels=true_labels, predicted_labels=adata.obs[cluster_col])
                results["ari"] = ari_score

            if metric == "asw":
                from pertpy.tools._perturbation_space._metrics import asw

                # TODO pass kwargs

                if 'distances' not in kwargs.keys():
                    distances = pairwise_distances(self.X, metric="euclidean")
                else:
                    distances = kwargs['distances']

                asw_score = asw(pairwise_distances=distances, labels=true_labels)
                results["asw"] = asw_score

        return results
