from __future__ import annotations

import warnings
from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
from pandas.errors import PerformanceWarning
from scipy.sparse import issparse, sparray

from pertpy.tools._perturbation_efficacy._base import PerturbationEfficacyAnalyzer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from anndata import AnnData


@singledispatch
def _subset_column_mean(matrix: np.ndarray, row_mask: np.ndarray) -> np.ndarray:
    """Per-column mean over the rows selected by `row_mask`."""
    return matrix[row_mask].mean(axis=0)


@singledispatch
def _project(matrix: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project each row onto `direction` (the unnormalized scalar projection numerator)."""
    return matrix @ direction


@singledispatch
def _leave_one_out_numerators(matrix: np.ndarray, numerator: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """For each gene (column) j, the projection numerator recomputed with gene j left out."""
    return numerator[:, None] - matrix * direction[None, :]


@_subset_column_mean.register(sparray)
def _(matrix, row_mask: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[row_mask].mean(axis=0)).ravel()


@_project.register(sparray)
def _(matrix, direction: np.ndarray) -> np.ndarray:
    return np.asarray(matrix @ direction).ravel()


@_leave_one_out_numerators.register(sparray)
def _(matrix, numerator: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return numerator[:, None] - matrix.multiply(direction[None, :]).toarray()


class Mixscale(PerturbationEfficacyAnalyzer):
    """Continuous perturbation scoring for pooled CRISPR screens.

    Where :class:`~pertpy.tools.Mixscape` assigns each cell a binary perturbed/non-perturbed label, Mixscale assigns a continuous perturbation score that reflects how strongly each cell responded.
    This is useful for CRISPRi/CRISPRa screens where cells show a gradient of responses rather than a clean knockout, and as input to downstream weighted differential expression and pathway analyses.

    The method is described in Jiang, Dalgarno et al., "Systematic reconstruction of molecular pathway signatures using scalable single-cell perturbation screens", Nature Cell Biology (2025) {cite}`Jiang2025`.
    It reproduces the reference implementation from the satijalab/Mixscale R package (https://github.com/satijalab/Mixscale).
    """

    def mixscale(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        new_class_name: str = "mixscale_score",
        layer: str | None = None,
        min_de_genes: int = 5,
        max_de_genes: int = 100,
        logfc_threshold: float = 0.25,
        de_layer: str | None = None,
        test_method: str = "wilcoxon",
        scale: bool = True,
        split_by: str | None = None,
        pval_cutoff: float = 5e-2,
        fine_mode: bool = False,
        fine_mode_labels: str = "guide_id",
        de_genes_by_target: Mapping[str, Sequence[str]] | None = None,
        harmonize: bool = False,
        harmonize_min_proportion: float = 0.1,
        random_state: int = 0,
        copy: bool = False,
    ):
        """Calculate a continuous perturbation score per cell with the Mixscale method.

        For every target gene the large-effect differentially expressed (DE) genes between its cells and the control cells are determined.
        The perturbation direction vector (mean perturbed minus mean control over those genes) is computed, and each cell's perturbation signature is projected onto that vector.
        The per-cell projection is then standardized against the control distribution.
        DE genes are detected on all cells pooled, while the direction vector and standardization are computed within each `split_by` group.
        The automatic DE detection relies on :func:`scanpy.tl.rank_genes_groups` and may select a slightly different gene set than the reference implementation; pass `de_genes_by_target` to score against a fixed gene set instead.

        Run :meth:`perturbation_signature` first to populate `.layers["X_pert"]`.

        Args:
            adata: The annotated data object.
            pert_key: The column of `.obs` with target gene labels.
            control: Control category from the `pert_key` column.
            new_class_name: Name of the score column to be stored in `.obs`.
            layer: Key from `adata.layers` whose value is used for scoring. If `None`, `.layers["X_pert"]` is used.
            min_de_genes: Required number of DE genes for scoring a perturbation. Perturbations with fewer DE genes are not scored and their cells receive the fallback score of 1.
            max_de_genes: Maximum number of top DE genes (by adjusted p-value) used for scoring.
            logfc_threshold: Minimum absolute log fold-change for a gene to be considered a large-effect DE gene.
            de_layer: Layer used for the DE test. If `None`, `adata.X` is used.
            test_method: Method passed to :func:`scanpy.tl.rank_genes_groups` for DE testing.
            scale: Whether to z-score each gene's perturbation signature (mean-centered and scaled to unit variance, then clipped at 10) before scoring.
            split_by: `.obs` column with a condition/cell-type annotation. The direction vector and standardization are computed separately within each group, while DE genes are still detected on all cells.
            pval_cutoff: Adjusted p-value cut-off for selecting significant DE genes.
            fine_mode: If `True`, DE genes are computed per gRNA (`fine_mode_labels`) and pooled per target gene, rather than once per target gene.
            fine_mode_labels: `.obs` column with gRNA identifiers, used when `fine_mode` is `True`.
            de_genes_by_target: Optional mapping from target gene to a user-defined list of DE genes. When given, the DE test is skipped entirely and targets absent from the mapping are not scored.
            harmonize: If `True` and `split_by` resolves to more than one group, control cells are subsampled so that their per-group composition matches the perturbed cells before the DE test.
            harmonize_min_proportion: Minimum fraction of control cells that must be retained during harmonization. Groups are dropped until the constraint is met.
            random_state: Seed for the control subsampling performed during harmonization.
            copy: Determines whether a copy of `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the scores in `.obs`.
            Otherwise, writes the scores directly to `.obs` of the provided `adata`.

            The following fields are added:

            - `adata.obs[new_class_name]`: Continuous perturbation score per cell. Control cells receive 0, cells of perturbations that could not be scored receive 1, and all other cells receive the projection standardized against the control distribution. Higher values indicate a stronger response.
            - `adata.uns["mixscale"]`: Per target gene and split, a :class:`~pandas.DataFrame` with the raw projection (`pvec`), the cell labels, and the leave-one-out projections (one column per DE gene).
            - `adata.uns["mixscale_de_genes"]`: The DE genes used for each target gene.

        Examples:
            Compute continuous perturbation scores:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms = pt.tl.Mixscale()
            >>> ms.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms.mixscale(mdata["rna"], "gene_target", "NT", layer="X_pert")
        """
        if copy:
            adata = adata.copy()

        if layer is not None:
            X = adata.layers[layer]
        else:
            try:
                X = adata.layers["X_pert"]
            except KeyError:
                raise KeyError(
                    "No 'X_pert' found in .layers! Please run perturbation_signature first to calculate the perturbation signature!"
                ) from None

        if split_by is None:
            split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = list(split_obs.unique())
            split_masks = [(split_obs == category).to_numpy() for category in categories]

        # DE genes are detected on all cells pooled (the direction vector and standardization are per split).
        perturbation_markers = self._get_mixscale_markers(
            adata,
            pert_key=pert_key,
            control=control,
            de_layer=de_layer,
            test_method=test_method,
            logfc_threshold=logfc_threshold,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            max_de_genes=max_de_genes,
            fine_mode=fine_mode,
            fine_mode_labels=fine_mode_labels,
            de_genes_by_target=de_genes_by_target,
            harmonize=harmonize,
            harmonize_min_proportion=harmonize_min_proportion,
            split_by=split_by,
            random_state=random_state,
        )

        var_loc = {name: i for i, name in enumerate(adata.var_names)}
        pert_values = adata.obs[pert_key].to_numpy()
        gv_list: dict[str, dict] = {}
        scores = np.zeros(adata.n_obs)
        scored = np.zeros(adata.n_obs, dtype=bool)

        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            nt_mask = (pert_values == control) & split_mask
            genes_in_split = set(pert_values[split_mask]).difference([control])
            for gene in genes_in_split:
                de_genes = [g for g in perturbation_markers.get(gene, ()) if g in var_loc]
                if len(de_genes) == 0:
                    continue

                guide_mask = (pert_values == gene) & split_mask
                all_mask = guide_mask | nt_mask
                guide_in_dat = guide_mask[all_mask]
                nt_in_dat = nt_mask[all_mask]
                if not guide_in_dat.any() or not nt_in_dat.any():
                    continue

                de_indices = [var_loc[g] for g in de_genes]
                dat = X[all_mask][:, de_indices].astype(np.float64)
                if scale:
                    dat = self._scale_features(dat)

                vec = _subset_column_mean(dat, guide_in_dat) - _subset_column_mean(dat, nt_in_dat)
                vec_norm_sq = float(vec @ vec)
                if not vec_norm_sq > 0:
                    continue

                numerator = _project(dat, vec)
                pvec = numerator / vec_norm_sq
                vec_sq = vec * vec
                with np.errstate(divide="ignore", invalid="ignore"):
                    loo = _leave_one_out_numerators(dat, numerator, vec) / (vec_norm_sq - vec_sq[None, :])

                nt_pvec = pvec[nt_in_dat]
                std_nt = nt_pvec.std(ddof=1)
                if std_nt == 0 or np.isnan(std_nt):
                    std_nt = 1.0
                guide_positions = np.flatnonzero(guide_mask)
                scores[guide_positions] = (pvec[guide_in_dat] - nt_pvec.mean()) / std_nt
                scored[guide_positions] = True

                all_names = adata.obs_names[all_mask]
                gv = pd.DataFrame(index=all_names)
                gv["pvec"] = pvec
                gv[pert_key] = control
                gv.loc[all_names[guide_in_dat], pert_key] = gene
                gv = pd.concat([gv, pd.DataFrame(loo, index=all_names, columns=de_genes)], axis=1)
                gv_list.setdefault(gene, {})[category] = gv

        scores[(~scored) & (pert_values != control)] = 1.0

        adata.obs[new_class_name] = scores
        adata.uns["mixscale"] = gv_list
        adata.uns["mixscale_de_genes"] = {gene: np.asarray(genes) for gene, genes in perturbation_markers.items()}

        if copy:
            return adata

    @staticmethod
    def _scale_features(dat, *, scale_max: float = 10.0) -> np.ndarray:
        """Z-score each gene (column) and clip, mirroring Seurat's `ScaleData`.

        Zero-centering necessarily densifies the (cells x DE-gene) submatrix, exactly as Seurat's `ScaleData` and :meth:`~pertpy.tools.Mixscape.mixscape` do.
        """
        dat = dat.toarray() if issparse(dat) else np.asarray(dat, dtype=np.float64)
        dat = dat.astype(np.float64, copy=False)
        mean = dat.mean(axis=0)
        std = dat.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        scaled = (dat - mean) / std
        np.clip(scaled, None, scale_max, out=scaled)
        return scaled

    def _get_mixscale_markers(
        self,
        adata: AnnData,
        *,
        pert_key: str,
        control: str,
        de_layer: str | None,
        test_method: str,
        logfc_threshold: float,
        pval_cutoff: float,
        min_de_genes: int,
        max_de_genes: int,
        fine_mode: bool,
        fine_mode_labels: str,
        de_genes_by_target: Mapping[str, Sequence[str]] | None,
        harmonize: bool,
        harmonize_min_proportion: float,
        split_by: str | None,
        random_state: int,
    ) -> dict[str, np.ndarray]:
        """Determine the large-effect DE genes for each target gene, pooling across all cells.

        Returns a mapping from target gene to the ordered array of DE gene names, empty when fewer than `min_de_genes` survive the filters.
        """
        var_names = set(adata.var_names)
        gene_targets = set(adata.obs[pert_key]).difference([control])
        nt_cells = adata.obs_names[adata.obs[pert_key] == control]
        markers: dict[str, np.ndarray] = {}

        for gene in gene_targets:
            if de_genes_by_target is not None:
                supplied = list(dict.fromkeys(g for g in de_genes_by_target.get(gene, ()) if g in var_names))
                if len(supplied) == 0:
                    warnings.warn(
                        f"No DE genes provided for perturbation {gene!r} in de_genes_by_target; it will not be scored.",
                        stacklevel=2,
                    )
                de_genes = np.array(supplied, dtype=object)
            else:
                guide_cells = adata.obs_names[adata.obs[pert_key] == gene]
                ref_cells = nt_cells
                if harmonize and split_by is not None:
                    ref_cells = self._harmonize_control_cells(
                        adata,
                        target_cells=guide_cells,
                        control_cells=nt_cells,
                        split_by=split_by,
                        min_proportion=harmonize_min_proportion,
                        random_state=random_state,
                    )

                if fine_mode:
                    pooled: list[str] = []
                    for guide in adata.obs.loc[guide_cells, fine_mode_labels].unique():
                        guide_subset = guide_cells[adata.obs.loc[guide_cells, fine_mode_labels] == guide]
                        for g in self._de_for_pair(
                            adata,
                            guide_subset,
                            ref_cells,
                            de_layer=de_layer,
                            test_method=test_method,
                            logfc_threshold=logfc_threshold,
                            pval_cutoff=pval_cutoff,
                        ):
                            if g not in pooled:
                                pooled.append(g)
                    de_genes = np.array(pooled, dtype=object)
                else:
                    de_genes = self._de_for_pair(
                        adata,
                        guide_cells,
                        ref_cells,
                        de_layer=de_layer,
                        test_method=test_method,
                        logfc_threshold=logfc_threshold,
                        pval_cutoff=pval_cutoff,
                    )

            if len(de_genes) > max_de_genes:
                de_genes = de_genes[:max_de_genes]
            if len(de_genes) < min_de_genes:
                de_genes = np.array([], dtype=object)
            markers[gene] = de_genes

        return markers

    @staticmethod
    def _de_for_pair(
        adata: AnnData,
        group_cells,
        reference_cells,
        *,
        de_layer: str | None,
        test_method: str,
        logfc_threshold: float,
        pval_cutoff: float,
    ) -> np.ndarray:
        """Wilcoxon-style DE between two cell sets; returns gene names passing the filters, sorted by raw p-value."""
        group_cells = pd.Index(group_cells)
        reference_cells = pd.Index(reference_cells)
        if len(group_cells) == 0 or len(reference_cells) == 0:
            return np.array([], dtype=object)

        sub = adata[adata.obs_names.isin(group_cells.union(reference_cells))].copy()
        groups = np.where(sub.obs_names.isin(group_cells), "perturbed", "control")
        sub.obs["_mixscale_de"] = pd.Categorical(groups, categories=["control", "perturbed"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", PerformanceWarning)
            sc.tl.rank_genes_groups(
                sub,
                layer=de_layer,
                groupby="_mixscale_de",
                groups=["perturbed"],
                reference="control",
                method=test_method,
                use_raw=False,
            )
        result = sub.uns["rank_genes_groups"]
        names = np.asarray(result["names"]["perturbed"])
        logfoldchanges = np.asarray(result["logfoldchanges"]["perturbed"])
        pvals = np.asarray(result["pvals"]["perturbed"])
        pvals_adj = np.asarray(result["pvals_adj"]["perturbed"])

        keep = (np.abs(logfoldchanges) >= logfc_threshold) & (pvals_adj < pval_cutoff)
        names, pvals = names[keep], pvals[keep]
        return names[np.argsort(pvals, kind="stable")]

    @staticmethod
    def _harmonize_control_cells(
        adata: AnnData,
        *,
        target_cells,
        control_cells,
        split_by: str,
        min_proportion: float,
        random_state: int,
    ) -> pd.Index:
        """Subsample control cells so their per-group composition matches the perturbed cells.

        The control subsampling uses NumPy's random generator and therefore does not reproduce the exact cells drawn by the Mixscale R package, but follows the same per-group counting logic.
        The R reference gates harmonization on the number of split columns, making it a no-op for a single split column; here it activates whenever the single split column has more than one group.
        """
        target_cells = pd.Index(target_cells)
        control_cells = pd.Index(control_cells)
        split = adata.obs[split_by]
        groups = list(split.unique())
        if len(groups) <= 1:
            return control_cells

        rng = np.random.default_rng(random_state)
        target_split = split.loc[target_cells]
        control_split = split.loc[control_cells]

        active_groups = list(groups)
        while True:
            n_target = np.array([(target_split == g).sum() for g in active_groups], dtype=float)
            n_control = np.array([(control_split == g).sum() for g in active_groups], dtype=float)
            if n_target.sum() == 0 or (n_target == 0).any() or (n_control == 0).any():
                # cannot harmonize cleanly; fall back to all control cells
                return control_cells
            prop_target = n_target / n_target.sum()
            total_desired = np.floor((n_control / prop_target).min())
            if total_desired >= min_proportion * n_control.sum():
                break
            del active_groups[int(np.argmin(n_control))]
            if len(active_groups) <= 1:
                return control_cells

        desired = np.floor(total_desired * prop_target).astype(int)
        sampled: list[str] = []
        for group, n_desired in zip(active_groups, desired, strict=True):
            pool = control_cells[(control_split == group).to_numpy()]
            n_draw = min(int(n_desired), len(pool))
            sampled.extend(rng.choice(np.asarray(pool), size=n_draw, replace=False).tolist())
        return pd.Index(sampled)
