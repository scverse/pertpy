from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import entropy

from pertpy._logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pertpy._types import RandomStateLike
    from pertpy.tools._distances._distances import Metric


def _sklearn_random_state(random_state: RandomStateLike) -> int | np.random.RandomState | None:
    """Normalize a random state to something scikit-learn accepts (an int, a ``RandomState`` or ``None``)."""
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(np.iinfo(np.int32).max))
    return random_state


def _resolve_matrix(adata: AnnData, *, layer_key: str | None, embedding_key: str | None) -> np.ndarray:
    """Pick the cell-by-feature matrix from a layer, an obsm embedding, or ``.X``.

    Layer wins over embedding; both default back to ``.X``; passing both raises.
    """
    if layer_key is not None and embedding_key is not None:
        raise ValueError("Please, select just either layer or embedding for computation.")
    if layer_key is not None:
        if layer_key not in adata.layers:
            raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
        return np.asarray(adata.layers[layer_key])
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
        return np.asarray(adata.obsm[embedding_key])
    return np.asarray(adata.X)


def _constant_obs_per_group(obs: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Collapse ``obs`` to one row per ``target_col`` value, keeping only columns constant within every group.

    Columns that vary within any group are dropped so the result can be safely mapped back onto a perturbation-level AnnData.
    """
    grouped = obs.groupby(target_col, observed=True).agg(
        lambda values: next(iter(set(values))) if len(set(values)) == 1 else np.nan
    )
    return grouped.loc[:, ~grouped.isna().any()]


def _carry_constant_obs(ps_adata: AnnData, source_obs: pd.DataFrame, target_col: str) -> None:
    """Copy every ``source_obs`` column that is constant within each ``target_col`` group onto ``ps_adata``."""
    extra = _constant_obs_per_group(source_obs, target_col)
    for col in extra.columns:
        if col == target_col:
            continue
        ps_adata.obs[col] = ps_adata.obs[target_col].map(extra[col].to_dict())


def _vector_distance(u: np.ndarray, v: np.ndarray, metric: str) -> float:
    """Distance between two 1D perturbation vectors."""
    if metric == "euclidean":
        return float(np.linalg.norm(u - v))
    if metric == "cosine":
        denom = float(np.linalg.norm(u) * np.linalg.norm(v))
        return 1.0 - float(np.dot(u, v)) / denom if denom else float("nan")
    if metric == "pearson":
        if u.std() == 0 or v.std() == 0:
            return float("nan")
        return 1.0 - float(np.corrcoef(u, v)[0, 1])
    raise ValueError(f"Unknown metric {metric!r}. Choose from 'euclidean', 'cosine', 'pearson'.")


def _subtract_control_mean(
    matrix: np.ndarray,
    control_mask: np.ndarray,
    group_masks: list[np.ndarray],
    *,
    name: str,
) -> np.ndarray:
    """Return ``matrix`` with the within-group control mean subtracted from every row.

    Groups with no control cells are left untouched and a warning is emitted.
    """
    out = np.zeros_like(matrix, dtype=float)
    for mask in group_masks:
        in_group = mask & control_mask
        n_control = int(in_group.sum())
        if n_control == 0:
            logger.warning(
                f"No control cells found for one group when computing {name!r}; "
                "leaving those rows unchanged (no control subtraction applied)."
            )
            out[mask, :] = matrix[mask, :]
            continue
        control_mean = matrix[in_group, :] if n_control == 1 else np.mean(matrix[in_group, :], axis=0)
        out[mask, :] = matrix[mask, :] - control_mean
    return out


class PerturbationSpace:
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single data points in an embeddings summarize a cell, whereas in a perturbation space, data points summarize whole perturbations.
    """

    def __init__(self):
        self.control_diff_computed = False

    def compute_control_diff(
        self,
        adata: AnnData,
        *,
        target_col: str = "perturbation",
        group_col: str | None = None,
        reference_key: str = "control",
        layer_key: str | None = None,
        new_layer_key: str = "control_diff",
        embedding_key: str | None = None,
        new_embedding_key: str = "control_diff",
        all_data: bool = False,
        copy: bool = True,
    ) -> AnnData:
        """Subtract mean of the control from the perturbation.

        Args:
            adata: Anndata object of size cells x genes.
            target_col: `.obs` column name that stores the label of the perturbation applied to each cell.
            group_col: `.obs` column name that stores the label of the group of each cell. If None, ignore groups.
            reference_key: The key of the control values.
            layer_key: Key of the AnnData layer to use for computation.
            new_layer_key: the results are stored in the given layer.
            embedding_key: `obsm` key of the AnnData embedding to use for computation.
            new_embedding_key: Results are stored in a new embedding in `obsm` with this key.
            all_data: if True, do the computation in all data representations (X, all layers and all embeddings)
            copy: If True returns a new AnnData; otherwise updates the input AnnData in place.

        Returns:
            Updated AnnData object.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> diff_adata = ps.compute_control_diff(mdata["rna"], target_col="gene_target", reference_key="NT")
        """
        if reference_key not in adata.obs[target_col].unique():
            raise ValueError(
                f"Reference key {reference_key} not found in {target_col}. {reference_key} must be in obs column {target_col}."
            )

        if embedding_key is not None and embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key {embedding_key} not found in obsm keys of the anndata.")

        if layer_key is not None and layer_key not in adata.layers:
            raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")

        if copy:
            adata = adata.copy()

        control_mask = (adata.obs[target_col] == reference_key).to_numpy()
        if group_col is None:
            group_masks: list[np.ndarray] = [np.ones(adata.n_obs, dtype=bool)]
        else:
            group_masks = [(adata.obs[group_col] == sample).to_numpy() for sample in adata.obs[group_col].unique()]

        if layer_key:
            adata.layers[new_layer_key] = _subtract_control_mean(
                adata.layers[layer_key], control_mask, group_masks, name=new_layer_key
            )

        if embedding_key:
            adata.obsm[new_embedding_key] = _subtract_control_mean(
                adata.obsm[embedding_key], control_mask, group_masks, name=new_embedding_key
            )

        if (not layer_key and not embedding_key) or all_data:
            adata.X = _subtract_control_mean(np.asarray(adata.X), control_mask, group_masks, name="X")

        if all_data:
            for local_layer_key in list(adata.layers.keys()):
                if local_layer_key in {layer_key, new_layer_key}:
                    continue
                new_key = local_layer_key + "_control_diff"
                adata.layers[new_key] = _subtract_control_mean(
                    adata.layers[local_layer_key], control_mask, group_masks, name=new_key
                )

            for local_embedding_key in list(adata.obsm.keys()):
                if local_embedding_key in {embedding_key, new_embedding_key}:
                    continue
                new_key = local_embedding_key + "_control_diff"
                adata.obsm[new_key] = _subtract_control_mean(
                    adata.obsm[local_embedding_key], control_mask, group_masks, name=new_key
                )

        self.control_diff_computed = True

        return adata

    def _combine(
        self,
        adata: AnnData,
        *,
        perturbations: Iterable[str],
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reference_key: str,
        new_pert_name: str,
        ensure_consistency: bool,
        target_col: str,
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Backend for both ``add`` and ``subtract``.

        ``op`` is applied as ``op(running_total, adata[perturbation].X)`` so callers pick addition or subtraction (or anything else operating in-place on a numpy array).
        """
        perturbations = list(perturbations)
        for perturbation in perturbations:
            if perturbation not in adata.obs_names:
                raise ValueError(
                    f"Perturbation {perturbation} not found in adata.obs_names. {perturbation} must be in adata.obs_names."
                )

        if ensure_consistency:
            adata = self.compute_control_diff(adata, copy=True, all_data=True, target_col=target_col)
        else:
            warnings.warn(
                "Combining perturbations without `ensure_consistency=True` is only well-defined "
                "if the input was already differenced against control "
                "(otherwise perturbation - perturbation != control).",
                stacklevel=3,
            )

        # sc.get.aggregate can leave a `None`-keyed layer behind (the pre-aggregation .X);
        # PseudobulkSpace.compute strips it but defensively re-strip here so callers passing
        # a hand-built AnnData don't crash inside the string ops below.
        layer_keys = [k for k in adata.layers if isinstance(k, str)]
        obsm_keys = [k for k in adata.obsm if isinstance(k, str)]
        rename_back = (
            {
                key: key.removesuffix("_control_diff")
                for key in [*layer_keys, *obsm_keys]
                if key.endswith("_control_diff")
            }
            if ensure_consistency
            else {}
        )

        def _running(values: np.ndarray) -> np.ndarray:
            result = values[adata.obs_names.get_loc(reference_key)].astype(float, copy=True)
            for perturbation in perturbations:
                op(result, values[adata.obs_names.get_loc(perturbation)])
            return result

        new_layers: dict[str, np.ndarray] = {}
        for layer_key in layer_keys:
            mat = adata.layers[layer_key]
            new_layers[rename_back.get(layer_key, layer_key)] = np.concatenate(
                (np.asarray(mat), _running(np.asarray(mat))[None, :]), axis=0
            )

        new_obsm: dict[str, np.ndarray] = {}
        for embedding_key in obsm_keys:
            mat = adata.obsm[embedding_key]
            new_obsm[rename_back.get(embedding_key, embedding_key)] = np.concatenate(
                (np.asarray(mat), _running(np.asarray(mat))[None, :]), axis=0
            )

        X = np.asarray(adata.X)
        new_X = np.concatenate((X, _running(X)[None, :]), axis=0)

        new_perturbation = AnnData(X=new_X)
        new_index = adata.obs_names.append(pd.Index([new_pert_name]))
        new_perturbation.obs_names = new_index
        new_perturbation.obs = adata.obs.reindex(new_index)

        for key, value in new_layers.items():
            new_perturbation.layers[key] = value
        for key, value in new_obsm.items():
            new_perturbation.obsm[key] = value

        new_perturbation.obs[target_col] = new_perturbation.obs_names.astype("category")

        if ensure_consistency:
            return new_perturbation, adata
        return new_perturbation

    def add(
        self,
        adata: AnnData,
        *,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = True,
        target_col: str = "perturbation",
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Add perturbations linearly. Assumes input of size n_perts x dimensionality.

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to add.
            reference_key: perturbation source from which the perturbation summation starts.
            ensure_consistency: If True, differentiate against control via `compute_control_diff` before combining so that "perturbation - perturbation == control" holds in the resulting space. Set False only if the input has already been differenced.
            target_col: `.obs` column name that stores the label of the perturbation applied to each cell.

        Returns:
            Anndata object of size (n_perts+1) x dim, where the last row is the addition of the specified perturbations.
            If ensure_consistency is True, returns a tuple of (new_perturbation, adata) where adata is the AnnData object provided as input but updated using compute_control_diff.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.add(ps_adata, perturbations=["ATF2", "CD86"], reference_key="NT")
        """
        perturbations = list(perturbations)
        new_pert_name = "+".join(perturbations)

        def _iadd(a: np.ndarray, b: np.ndarray) -> None:
            a += b

        return self._combine(
            adata,
            perturbations=perturbations,
            op=_iadd,
            reference_key=reference_key,
            new_pert_name=new_pert_name,
            ensure_consistency=ensure_consistency,
            target_col=target_col,
        )

    def subtract(
        self,
        adata: AnnData,
        *,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = True,
        target_col: str = "perturbation",
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Subtract perturbations linearly. Assumes input of size n_perts x dimensionality.

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to subtract.
            reference_key: Perturbation source from which the perturbation subtraction starts.
            ensure_consistency: If True, differentiate against control via `compute_control_diff` before combining so that "perturbation - perturbation == control" holds in the resulting space. Set False only if the input has already been differenced.
            target_col: `.obs` column name that stores the label of the perturbation applied to each cell.

        Returns:
            Anndata object of size (n_perts+1) x dim, where the last row is the subtraction of the specified perturbations.
            If ensure_consistency is True, returns a tuple of (new_perturbation, adata) where adata is the AnnData object provided as input but updated using compute_control_diff.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.subtract(ps_adata, reference_key="ATF2", perturbations=["BRD4", "CUL3"])
        """
        perturbations = list(perturbations)
        new_pert_name = reference_key + "-" + "-".join(perturbations)

        def _isub(a: np.ndarray, b: np.ndarray) -> None:
            a -= b

        return self._combine(
            adata,
            perturbations=perturbations,
            op=_isub,
            reference_key=reference_key,
            new_pert_name=new_pert_name,
            ensure_consistency=ensure_consistency,
            target_col=target_col,
        )

    def label_transfer(
        self,
        adata: AnnData,
        *,
        target_column: str = "perturbation",
        column_uncertainty_score_key: str = "perturbation_transfer_uncertainty",
        target_val: str = "unknown",
        neighbors_key: str = "neighbors",
    ) -> None:
        """Impute missing values in the specified column using KNN imputation in the space defined by `use_rep`.

        Uncertainty is calculated as the entropy of the label distribution in the neighborhood of the target cell.
        In other words, a cell where all neighbors have the same set of labels will have an uncertainty of 0, whereas a cell where all neighbors have many different labels will have high uncertainty.

        Args:
            adata: The AnnData object containing single-cell data.
            target_column: The column name in adata.obs to perform imputation on.
            column_uncertainty_score_key: The column name in adata.obs to store the uncertainty score of the label transfer.
            target_val: The target value to impute.
            neighbors_key: The key in adata.uns where the neighbors are stored.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> import numpy as np
            >>> adata = sc.datasets.pbmc68k_reduced()
            >>> # randomly dropout 10% of the data annotations
            >>> adata.obs["perturbation"] = adata.obs["louvain"].astype(str).copy()
            >>> random_cells = np.random.choice(adata.obs.index, int(adata.obs.shape[0] * 0.1), replace=False)
            >>> adata.obs.loc[random_cells, "perturbation"] = "unknown"
            >>> sc.pp.neighbors(adata)
            >>> sc.tl.umap(adata)
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps.label_transfer(adata)
        """
        if neighbors_key not in adata.uns:
            raise ValueError(f"Key {neighbors_key} not found in adata.uns. Please run `sc.pp.neighbors` first.")

        labels = adata.obs[target_column].astype(str)
        target_cells = labels == target_val

        connectivities = adata.obsp[adata.uns[neighbors_key]["connectivities_key"]]
        # convert labels to an incidence matrix
        one_hot_encoded_labels = adata.obs[target_column].astype(str).str.get_dummies()
        # convert to distance-weighted neighborhood incidence matrix
        weighted_label_occurence = pd.DataFrame(
            (one_hot_encoded_labels.values.T * connectivities).T,
            index=adata.obs_names,
            columns=one_hot_encoded_labels.columns,
        )
        # choose best label for each target cell
        best_labels = weighted_label_occurence.drop(target_val, axis=1)[target_cells].idxmax(axis=1)
        adata.obs[target_column] = labels
        adata.obs.loc[target_cells, target_column] = best_labels

        # calculate uncertainty
        uncertainty = np.zeros(adata.n_obs)
        uncertainty[target_cells] = entropy(weighted_label_occurence.drop(target_val, axis=1)[target_cells], axis=1)
        adata.obs[column_uncertainty_score_key] = uncertainty

    def nearest_perturbations(
        self,
        adata: AnnData,
        perturbation: str,
        *,
        target_col: str = "perturbation",
        n_neighbors: int = 10,
        layer_key: str | None = None,
        embedding_key: str | None = None,
        metric: str = "euclidean",
    ) -> pd.DataFrame:
        """Rank perturbations by their proximity to a query perturbation in a perturbation space.

        Operates on a perturbation-level AnnData (one observation per perturbation), i.e. the output of any ``compute``.
        Useful for discovering perturbations with a similar mechanism of action.
        If ``adata.obsp["distances"]`` is present (as produced by :class:`~pertpy.tools.DistanceSpace`) and no representation is requested explicitly, those precomputed distances are used directly.

        Args:
            adata: Perturbation-level AnnData indexed by perturbation.
            perturbation: The query perturbation to find neighbors for.
            target_col: `.obs` column identifying each perturbation.
            n_neighbors: Number of nearest perturbations to return.
            layer_key: Layer to compute distances from.
            embedding_key: `.obsm` embedding to compute distances from.
            metric: Distance metric passed to :func:`sklearn.metrics.pairwise_distances`.

        Returns:
            DataFrame indexed by perturbation with a ``distance`` column, sorted ascending and excluding the query.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.norman_2019()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(adata, target_col="perturbation_name")
            >>> neighbors = ps.nearest_perturbations(ps_adata, "CBL+CNN1", target_col="perturbation_name")
        """
        names = (
            adata.obs[target_col].astype(str).to_numpy()
            if target_col in adata.obs
            else adata.obs_names.to_numpy().astype(str)
        )
        matches = np.flatnonzero(names == str(perturbation))
        if matches.size == 0:
            raise ValueError(f"Perturbation {perturbation!r} not found in adata.obs[{target_col!r}].")
        query_idx = int(matches[0])

        if layer_key is None and embedding_key is None and "distances" in adata.obsp:
            distances = np.asarray(adata.obsp["distances"])[query_idx]
        else:
            from sklearn.metrics import pairwise_distances

            coords = _resolve_matrix(adata, layer_key=layer_key, embedding_key=embedding_key)
            distances = pairwise_distances(coords[[query_idx]], coords, metric=metric)[0]

        keep = np.arange(len(names)) != query_idx
        result = pd.DataFrame({"distance": distances[keep]}, index=names[keep])
        return result.sort_values("distance").head(n_neighbors)

    def evaluate_combinations(
        self,
        adata: AnnData,
        *,
        combinations: Iterable[str] | None = None,
        target_col: str = "perturbation",
        reference_key: str = "control",
        metric: Literal["pearson", "cosine", "euclidean"] = "pearson",
        sep: str = "+",
    ) -> pd.DataFrame:
        """Score how well an additive model predicts combination perturbations.

        For every combination ``"A+B"`` whose components ``A`` and ``B`` are each present as single perturbations, the additive prediction ``effect(A) + effect(B)`` is compared against the measured combination effect, where effects are taken relative to ``reference_key``.
        A small distance indicates additive, non-interacting perturbations, whereas a large deviation flags a genetic or pharmacological interaction.

        Args:
            adata: Perturbation-level AnnData indexed by perturbation (one observation per perturbation).
            combinations: Combination names to evaluate. If None, every ``obs_name`` containing ``sep`` whose components are all present as singles is used.
            target_col: `.obs` column identifying each perturbation.
            reference_key: Control perturbation subtracted to obtain effects. If absent, values are used as-is.
            metric: Distance between predicted and measured combination effects.
            sep: Separator between components in combination names.

        Returns:
            DataFrame indexed by combination with ``distance``, ``predicted_magnitude`` and ``measured_magnitude`` columns, sorted by ascending distance.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.norman_2019()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(adata, target_col="perturbation_name")
            >>> scores = ps.evaluate_combinations(ps_adata, target_col="perturbation_name", reference_key="control")
        """
        names = adata.obs_names.astype(str)
        matrix = np.asarray(adata.X, dtype=float)
        vectors = {name: matrix[i] for i, name in enumerate(names)}

        effects = (
            {name: vec - vectors[reference_key] for name, vec in vectors.items()}
            if reference_key in vectors
            else vectors
        )

        if combinations is None:
            combinations = [n for n in names if sep in n and all(part in vectors for part in n.split(sep))]
        combinations = list(combinations)
        if not combinations:
            raise ValueError("No evaluable combinations found; provide `combinations` explicitly or check `sep`.")

        rows: dict[str, dict[str, float]] = {}
        for combo in combinations:
            if combo not in effects:
                raise ValueError(f"Combination {combo!r} not found in adata.")
            components = combo.split(sep)
            missing = [part for part in components if part not in effects]
            if missing:
                raise ValueError(f"Components {missing} of {combo!r} are not present as single perturbations.")
            predicted = np.sum([effects[part] for part in components], axis=0)
            measured = effects[combo]
            rows[combo] = {
                "distance": _vector_distance(predicted, measured, metric),
                "predicted_magnitude": float(np.linalg.norm(predicted)),
                "measured_magnitude": float(np.linalg.norm(measured)),
            }

        return pd.DataFrame.from_dict(rows, orient="index").sort_values("distance")

    def dose_response(
        self,
        adata: AnnData,
        *,
        target_col: str = "perturbation",
        dose_col: str = "dose",
        reference_key: str = "control",
        metric: Metric = "edistance",
        layer_key: str | None = None,
        embedding_key: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Quantify the effect size of each perturbation as a function of dose.

        For every (perturbation, dose) group the statistical distance to ``reference_key`` is computed in the chosen representation using :class:`~pertpy.tools.Distance`.
        Operates on cell-level data, since distances are defined between groups of cells.

        Args:
            adata: Cell-level AnnData.
            target_col: `.obs` column with the perturbation label.
            dose_col: `.obs` column with the (numeric) dose.
            reference_key: Control perturbation all doses are compared against.
            metric: Distance metric passed to :class:`~pertpy.tools.Distance`.
            layer_key: Layer to compute distances from.
            embedding_key: `.obsm` embedding to compute distances from.
            kwargs: Passed to :meth:`~pertpy.tools.Distance.onesided_distances`.

        Returns:
            Tidy DataFrame with ``perturbation``, ``dose`` and ``distance`` columns, sorted by perturbation then dose.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.srivatsan_2020_sciplex2()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> curves = ps.dose_response(adata, dose_col="dose_value", embedding_key="X_pca")
        """
        for col in (target_col, dose_col):
            if col not in adata.obs:
                raise ValueError(f"Column {col!r} does not exist in the .obs attribute.")
        from pertpy.tools._distances._distances import Distance

        sep = "\x1f"
        is_control = (adata.obs[target_col] == reference_key).to_numpy()
        group = adata.obs[target_col].astype(str).str.cat(adata.obs[dose_col].astype(str), sep=sep)
        group = group.mask(is_control, reference_key)
        grouped = adata.copy()
        grouped.obs["_dose_group"] = pd.Categorical(group)

        distance = Distance(metric=metric, layer_key=layer_key, obsm_key=embedding_key)
        dists = distance.onesided_distances(
            grouped, groupby="_dose_group", selected_group=reference_key, show_progressbar=False, **kwargs
        )
        if isinstance(dists, tuple):
            dists = dists[0]

        records = []
        for label, value in dists.items():
            if label == reference_key:
                continue
            perturbation, _, dose = str(label).partition(sep)
            records.append({"perturbation": perturbation, "dose": dose, "distance": float(value)})
        result = pd.DataFrame.from_records(records)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.suppress(ValueError, TypeError):
                result["dose"] = pd.to_numeric(result["dose"])
        return result.sort_values(["perturbation", "dose"]).reset_index(drop=True)

    def plot_similarity(  # pragma: no cover
        self,
        adata: AnnData,
        *,
        target_col: str = "perturbation",
        layer_key: str | None = None,
        embedding_key: str | None = None,
        metric: str = "euclidean",
        cmap: str = "viridis",
        **kwargs,
    ):
        """Plot a clustered heatmap of pairwise distances between perturbations.

        Uses ``adata.obsp["distances"]`` when present (e.g. from :class:`~pertpy.tools.DistanceSpace`) and otherwise computes pairwise distances in the chosen representation.

        Args:
            adata: Perturbation-level AnnData indexed by perturbation.
            target_col: `.obs` column identifying each perturbation.
            layer_key: Layer to compute distances from.
            embedding_key: `.obsm` embedding to compute distances from.
            metric: Distance metric passed to :func:`sklearn.metrics.pairwise_distances`.
            cmap: Matplotlib colormap.
            kwargs: Passed to :func:`seaborn.clustermap`.

        Returns:
            The :class:`seaborn.matrix.ClusterGrid` instance.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.norman_2019()
            >>> ds = pt.tl.DistanceSpace()
            >>> ds_adata = ds.compute(adata, target_col="perturbation_name", metric="edistance")
            >>> ds.plot_similarity(ds_adata, target_col="perturbation_name")
        """
        import seaborn as sns

        names = (
            adata.obs[target_col].astype(str).to_numpy()
            if target_col in adata.obs
            else adata.obs_names.to_numpy().astype(str)
        )
        if layer_key is None and embedding_key is None and "distances" in adata.obsp:
            distances = np.asarray(adata.obsp["distances"])
        else:
            from sklearn.metrics import pairwise_distances

            coords = _resolve_matrix(adata, layer_key=layer_key, embedding_key=embedding_key)
            distances = pairwise_distances(coords, metric=metric)
        frame = pd.DataFrame(distances, index=names, columns=names)

        return sns.clustermap(frame, cmap=cmap, **kwargs)
