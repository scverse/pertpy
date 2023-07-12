from typing import List, Literal

from anndata import AnnData
from sklearn.metrics import pairwise_distances

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


class ClusteringSpace(PerturbationSpace):
    """Applies various clustering techniques to an embedding."""

    def __init__(
        self, method: Literal["dendrogram", "k-means", "dbscan", "spectral clustering", "gaussian mixture"] = "k-means"
    ):
        """Initialize ClusteringSpace class.

        Args:
            method: ClusteringSpace method to use. Defaults to 'K-Means'.
        """

        if method.lower() == "k-means":
            from pertpy.tools._perturbation_space._simple import KMeansSpace

            method_fct = KMeansSpace()  # type: ignore
        elif method.lower() == "dbscan":
            from pertpy.tools._perturbation_space._simple import DBScanSpace

            method_fct = DBScanSpace()  # type: ignore
        elif method.lower() == "pseudobulk":
            from pertpy.tools._perturbation_space._simple import PseudobulkSpace

            method_fct = PseudobulkSpace()  # type: ignore
        elif method.lower() == "classifier":
            from pertpy.tools._perturbation_space._discriminator_classifier import DiscriminatorClassifierSpace

            method_fct = DiscriminatorClassifierSpace()  # type: ignore
        else:
            raise ValueError(f"Method {method} not recognized.")
        self.method_fct = method_fct

    def __call__(  # type: ignore
        self,
        adata: AnnData,
        *,
        reference_key: str = "control",
        layer_key: str = None,
        new_layer_key: str = None,
        embedding_key: str = None,
        new_embedding_key: str = None,
        copy: bool = False,
        **kwargs,
    ) -> AnnData:
        """Compute clustering space.

        Args:
            adata: The AnnData object that contains the data.
            layer_key: The layer of which to use the transcription values for to determine the perturbation space.
            embedding_key: The obsm matrix of which to use the embedding values for to determine the perturbation space.

        Returns:
            The determined perturbation space inside an AnnData object.
            # TODO add more detail

        Examples:
            >>> import pertpy as pt

            >>> adata = pt.dt.papalexi_2021()["rna"]
            >>> clsp = pt.tl.ClusteringSpace(method='K-Means')
            >>> perturbation_space = clsp()
            >>> TODO add something here
        """
        method_kwargs = {
            "reference_key": reference_key,
            "layer_key": layer_key,
            "new_layer_key": new_layer_key,
            "embedding_key": embedding_key,
            "new_embedding_key": new_embedding_key,
            "copy": copy,
            **kwargs,
        }

        return self.method_fct(adata, **method_kwargs)  # type: ignore

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

                distances = pairwise_distances(self.method_fct.X, metric="euclidean")

                asw_score = asw(pairwise_distances=distances, labels=true_labels)
                results["asw"] = asw_score

        return results
