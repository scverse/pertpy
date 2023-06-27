from typing import Literal

from anndata import AnnData


class PerturbationSpace:
    """Implements various ways of summarizing perturbations into a condensed space.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single dota points in an embeddings summarize a cell,
    whereas in a perturbation space, data points summarize whole perturbations.

    The method to calculate a perturbation space can be specified by a user.
    This class also provides various methods to evaluate perturbation spaces
    and to perform arithmetics with it.

    Currently available methods:
    # TODO all others with lots of detail
    - "pseudobulk": Pseudobulk based.
        Determines the pseudobulks of the conditions.

    Attributes:
        method: Name of perturbation space function.

    Examples:
        >>> import pertpy as pt

        >>> adata = pt.dt.papalexi_2021()["rna"]
        >>> pbps = pt.tl.PerturbationSpace(method='pseudobulk')
        >>> perturbation_space = pbps()
        >>> TODO add something here
    """

    def __init__(
        self,
        method: Literal["differential", "centroid", "pseudobulk"] = "pseudobulk",
    ):
        """Initialize PerturbationSpace class.

        Args:
            adata: The AnnData object to calculate the perturbation space for.
            method: PerturbationSpace method to use. Defaults to 'pseudobulk'.
        """
        if method == "differential":
            from pertpy.tools._perturbation_space._simple import DifferentialSpace

            method_fct = DifferentialSpace()
        elif method == "centroid":
            from pertpy.tools._perturbation_space._simple import CentroidSpace

            method_fct = CentroidSpace()  # type: ignore
        elif method == "pseudobulk":
            from pertpy.tools._perturbation_space._simple import PseudobulkSpace

            method_fct = PseudobulkSpace()  # type: ignore
        else:
            raise ValueError(f"Method {method} not recognized.")
        self.method_fct = method_fct

    def __call__(
        self,
        adata: AnnData,
        *,
        reference_key: str = "control",
        target_col: str = "perturbations",
        layer_key: str = None,
        new_layer_key: str = None,
        embedding_key: str = None,
        new_embedding_key: str = None,
        copy: bool = False,
        **kwargs,
    ) -> AnnData:
        """Compute perturbation space.

        Args:
            adata: The AnnData object to compute the perturbation space for.
            reference_key: The reference sample. Defaults to 'control'.
            target_col: The column containing the sample states (perturbations). Defaults to 'perturbations'.
            layer_key: The layer of which to use the transcription values for to determine the perturbation space.
            new_layer_key: The name of the new layer to add to the AnnData object.
            embedding_key: The obsm matrix of which to use the embedding values for to determine the perturbation space.
            new_embedding_key: The name of the new obsm key to add to the AnnData object.

        Returns:
            The determined perturbation space inside an AnnData object.
            # TODO add more detail

        Examples:
            >>> import pertpy as pt

            >>> adata = pt.dt.papalexi_2021()["rna"]
            >>> pbps = pt.tl.PerturbationSpace(method='pseudobulk')
            >>> perturbation_space = pbps()
            >>> TODO add something here
        """
        method_kwargs = {
            "reference_key": reference_key,
            "target_col": target_col,
            "layer_key": layer_key,
            "new_layer_key": new_layer_key,
            "embedding_key": embedding_key,
            "new_embedding_key": new_embedding_key,
            "copy": copy,
            **kwargs,
        }

        return self.method_fct(adata, **method_kwargs)  # type: ignore
