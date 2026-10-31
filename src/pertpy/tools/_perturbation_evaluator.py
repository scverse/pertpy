from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from pertpy._types import cast_frame, cast_matrix
from pertpy.tools._distances._distances import Distance

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from anndata import AnnData

    from pertpy._types import CSBase

Metric = Literal["mse", "edistance", "delta_pearson", "direction_accuracy", "top_k_overlap"]
Baseline = Literal["control_mean", "additive"]
Group = tuple[str, ...]


def _identifiers(adata: AnnData) -> None:
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Observation and feature identifiers must be unique.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("AnnData must contain observations and features.")


def _digest(values: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), ensure_ascii=False).encode()).hexdigest()


class PerturbationEvaluator:
    """Compare perturbation predictions with baselines fitted only on training cells.

    Features are aligned by ``var_names``; all inputs must have exactly the same
    feature set and measurement scale. No normalization or log transformation is
    performed. Cells are compared as groups, not paired observations.

    Args:
        perturbation_key: Column of ``obs`` identifying interventions.
        control: Label identifying unperturbed cells.
        context_key: Optional ``obs`` column (for example cell type). Groups and
            baselines are conditioned on this column; contexts are never pooled.
        layer_key: Matrix in ``layers`` to use for every input, or ``X`` if None.

    Examples:
        >>> evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl")
        >>> train, test = evaluator.split(
        ...     adata, holdout=["A+B"], strategy="combination", components={"A+B": ["A", "B"]}
        ... )
        >>> scores = evaluator.evaluate(test, {"model": predicted}, train=train, components={"A+B": ["A", "B"]})
    """

    def __init__(
        self,
        perturbation_key: str,
        control: str,
        *,
        context_key: str | None = None,
        layer_key: str | None = None,
    ) -> None:
        if not perturbation_key or not isinstance(control, str) or not control:
            raise ValueError("Provide a perturbation column and a nonempty string control label.")
        if context_key is not None and (not isinstance(context_key, str) or not context_key):
            raise ValueError("context_key must be a nonempty column name or None.")
        if context_key == perturbation_key:
            raise ValueError("Context and perturbation columns must be different.")
        self.perturbation_key = perturbation_key
        self.control = control
        self.context_key = context_key
        self.layer_key = layer_key

    @property
    def _columns(self) -> list[str]:
        return [self.context_key, self.perturbation_key] if self.context_key else [self.perturbation_key]

    def _groups(self, adata: AnnData) -> dict[Group, np.ndarray]:
        _identifiers(adata)
        obs = cast_frame(adata.obs)
        for col in self._columns:
            if col not in obs:
                raise ValueError(f"Missing observation column {col!r}.")
            if obs[col].isna().any() or not all(isinstance(x, str) and x for x in obs[col]):
                raise ValueError(f"Column {col!r} must contain nonempty string labels without missing values.")
        groups: dict[Group, list[int]] = {}
        for pos, row in enumerate(obs[self._columns].itertuples(index=False, name=None)):
            groups.setdefault(row, []).append(pos)
        return {key: np.asarray(groups[key], dtype=int) for key in sorted(groups)}

    def _matrix(self, adata: AnnData, features: pd.Index) -> np.ndarray | CSBase:
        if len(features) != adata.n_vars or set(features) != set(adata.var_names):
            raise ValueError("Inputs must have exactly the same feature identifiers; align or select them explicitly.")
        if self.layer_key is None:
            matrix = adata.X
        else:
            if self.layer_key not in adata.layers:
                raise ValueError(f"Missing layer {self.layer_key!r}.")
            matrix = adata.layers[self.layer_key]
        if not isinstance(matrix, np.ndarray) and not sparse.issparse(matrix):
            raise TypeError("Use an in-memory NumPy or SciPy sparse matrix; load backed or lazy data explicitly.")
        matrix = cast_matrix(matrix)
        if matrix.dtype.kind not in "biuf":
            raise ValueError("Measurements must be real numeric values.")
        if isinstance(matrix, np.ndarray):
            values = matrix
        else:
            matrix = matrix.tocsr()
            values = matrix.data
        if not np.isfinite(values).all():
            raise ValueError("Measurements must be finite; missing values are not biological zeros.")
        return matrix[:, adata.var_names.get_indexer(features)]

    def split(
        self,
        adata: AnnData,
        *,
        holdout: Sequence[str],
        strategy: Literal["perturbation", "combination", "context"] = "perturbation",
        components: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[AnnData, AnnData]:
        """Create explicit group holdouts without randomly splitting cells of an intervention.

        Args:
            adata: Data with globally unique cell identifiers.
            holdout: Intervention labels, or context labels for ``strategy='context'``.
            strategy: Hold out entire perturbations, entire combinations, or entire
                contexts. Control cells stay in training for intervention holdouts.
                A context holdout includes that context's control cells in test.
                Perturbation holdout uses exact labels, not every combination
                containing a target. Combination holdout also removes aliases
                with the same component set declared in ``components``.
            components: For combination holdouts, map each held-out label to its
                distinct single-intervention labels. Labels are never parsed implicitly.

        Returns:
            Independent training and test copies. Both include the same versioned
            ``uns['pertpy_evaluation_split']`` record with exact cell membership.
            Context holdouts may have unavailable context-matched baselines;
            :meth:`evaluate` reports those explicitly instead of using test controls.
        """
        self._groups(adata)
        if strategy not in ("perturbation", "combination", "context"):
            raise ValueError("Unknown split strategy.")
        if isinstance(holdout, str) or len(holdout) == 0 or len(set(holdout)) != len(holdout):
            raise ValueError("holdout must be a nonempty sequence of distinct labels.")
        if strategy == "context" and self.context_key is None:
            raise ValueError("A context split requires context_key.")
        col = self.context_key if strategy == "context" else self.perturbation_key
        labels = cast_frame(adata.obs)[col]
        if set(holdout) - set(labels):
            raise ValueError("Every held-out label must occur in the data.")
        if strategy != "context" and self.control in holdout:
            raise ValueError("The control cannot be a held-out perturbation.")
        test_mask = labels.isin(holdout).to_numpy()
        if strategy == "combination":
            held_out_components = set()
            for label in holdout:
                parts = self._components(label, components)
                if len(parts) < 2:
                    raise ValueError("Each held-out combination needs at least two distinct component labels.")
                held_out_components.add(frozenset(parts))
            # Known aliases such as A+B and B+A describe the same split unit.
            aliases = [
                label for label in set(labels) if frozenset(self._components(label, components)) in held_out_components
            ]
            test_mask = labels.isin(aliases).to_numpy()
        if test_mask.all() or not test_mask.any():
            raise ValueError("Both training and test partitions must be nonempty.")
        train, test = adata[~test_mask].copy(), adata[test_mask].copy()
        manifest = {
            "schema_version": 1,
            "strategy": strategy,
            "column": col,
            "holdout": sorted(holdout),
            "train_obs_names": train.obs_names.tolist(),
            "test_obs_names": test.obs_names.tolist(),
        }
        train.uns["pertpy_evaluation_split"] = json.loads(json.dumps(manifest))
        test.uns["pertpy_evaluation_split"] = json.loads(json.dumps(manifest))
        return train, test

    def _components(self, label: str, components: Mapping[str, Sequence[str]] | None) -> tuple[str, ...]:
        if components is None or label not in components:
            return ()
        parts = components[label]
        if (
            isinstance(parts, str)
            or len(parts) == 0
            or any(not isinstance(x, str) or not x or x == self.control for x in parts)
            or len(set(parts)) != len(parts)
            or label in parts
        ):
            raise ValueError("Components must be distinct non-control intervention labels.")
        if any(part in components and len(components[part]) > 1 for part in parts):
            raise ValueError("Components must name single interventions, not nested combinations.")
        return tuple(parts)

    def _baseline(
        self,
        name: Baseline,
        group: Group,
        train_matrix,
        train_groups: dict[Group, np.ndarray],
        components: Mapping[str, Sequence[str]] | None,
    ) -> tuple[np.ndarray | None, str]:
        control_group = (*group[:-1], self.control)
        if control_group not in train_groups:
            return None, "missing_training_control"
        control_mean = np.asarray(train_matrix[train_groups[control_group]].astype(np.float64).mean(axis=0)).reshape(
            1, -1
        )
        if name == "control_mean":
            return control_mean, "ok"
        parts = self._components(group[-1], components)
        if not parts:
            return None, "missing_components"
        component_groups = [(*group[:-1], part) for part in parts]
        if any(key not in train_groups for key in component_groups):
            return None, "missing_training_component"
        prediction = control_mean.copy()
        for key in component_groups:
            prediction += (
                np.asarray(train_matrix[train_groups[key]].astype(np.float64).mean(axis=0)).reshape(1, -1)
                - control_mean
            )
        if not np.isfinite(prediction).all():
            return None, "nonfinite_baseline"
        return prediction, "ok"

    @staticmethod
    def _scores(
        real: np.ndarray,
        predicted: np.ndarray,
        reference: np.ndarray | None,
        genes: np.ndarray,
        *,
        metrics: Sequence[Metric],
        top_k: int,
        effect_threshold: float,
    ) -> dict[str, tuple[float, str]]:
        scores: dict[str, tuple[float, str]] = {}
        actual_mean, predicted_mean = real.mean(axis=0), predicted.mean(axis=0)
        for metric in metrics:
            if metric in ("mse", "edistance"):
                scores[metric] = (float(Distance(metric)(real, predicted)), "ok")
                continue
            if reference is None:
                scores[metric] = (np.nan, "missing_reference_control")
                continue
            actual, estimated = actual_mean - reference, predicted_mean - reference
            if not np.isfinite(actual).all() or not np.isfinite(estimated).all():
                scores[metric] = (np.nan, "nonfinite_result")
                continue
            if metric == "delta_pearson":
                # Correlation is invariant to positive rescaling. Scale before
                # centering to avoid overflow in otherwise finite deltas/norms.
                a_scale, b_scale = np.abs(actual).max(), np.abs(estimated).max()
                a = actual / a_scale if a_scale > 0 else actual
                b = estimated / b_scale if b_scale > 0 else estimated
                a, b = a - a.mean(), b - b.mean()
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                scores[metric] = (
                    (float(np.clip(np.dot(a, b) / denominator, -1, 1)), "ok")
                    if denominator > 0
                    else (np.nan, "undefined_constant")
                )
            elif metric == "direction_accuracy":
                changed = np.abs(actual) > effect_threshold
                value = (
                    float(np.mean(np.sign(actual[changed]) == np.sign(estimated[changed]))) if changed.any() else np.nan
                )
                scores[metric] = (value, "ok" if changed.any() else "no_reference_effect")
            else:
                # Tie order depends on identifiers, never on the input feature order.
                k = min(top_k, len(genes))
                truth_top = np.lexsort((genes, -np.abs(actual)))[:k]
                prediction_top = np.lexsort((genes, -np.abs(estimated)))[:k]
                if np.abs(actual).max() <= effect_threshold or np.abs(estimated).max() <= effect_threshold:
                    scores[metric] = (np.nan, "no_rankable_effect")
                else:
                    scores[metric] = (float(len(set(truth_top) & set(prediction_top)) / k), "ok")
        return {
            key: (np.nan, "nonfinite_result") if status == "ok" and not np.isfinite(value) else (value, status)
            for key, (value, status) in scores.items()
        }

    def evaluate(
        self,
        ground_truth: AnnData,
        predictions: Mapping[str, AnnData],
        *,
        train: AnnData,
        components: Mapping[str, Sequence[str]] | None = None,
        baselines: Sequence[Baseline] = ("control_mean", "additive"),
        metrics: Sequence[Metric] = ("mse", "edistance", "delta_pearson", "direction_accuracy", "top_k_overlap"),
        feature_sets: Mapping[Group, Sequence[str]] | None = None,
        top_k: int = 20,
        effect_threshold: float = 0.0,
        split_name: str = "test",
    ) -> pd.DataFrame:
        """Return one row per model, test group, feature scope and metric.

        Args:
            ground_truth: Held-out measured cells. Control rows, if present, are
                excluded from scoring. Only training controls define the reference
                for changes in expression, including for external predictions.
            predictions: Named AnnData predictions, grouped by the same labels.
                Cell counts may differ. Missing prediction groups produce explicit
                unavailable rows. Unknown non-control groups are rejected.
            train: Training cells, disjoint from ground truth by cell identifier.
                Baselines use only this data. This checks declared membership, not
                an external model's training history or upstream preprocessing.
            components: Map a combination label to its single-intervention labels.
                The additive point prediction is the training control mean plus
                the sum of the component mean changes in the same context/scale.
            baselines: Include control-mean and additive point predictions. Missing
                training components/controls are reported; no fallback is hidden.
            metrics: ``mse`` and ``edistance`` reuse :class:`Distance`. Delta Pearson
                correlates gene-wise changes relative to the reference control.
                Direction accuracy considers only reference changes larger than
                ``effect_threshold``. Top-k overlap ranks absolute mean changes;
                it is not a statistical differential-expression test.
            feature_sets: Optional predeclared genes for each group tuple, ordered
                as ``(context, perturbation)`` or ``(perturbation,)``. Adds a
                ``selected`` scope alongside ``all``; this tool does not discover DEGs.
            top_k: Number of genes ranked for overlap, capped at the scope's size.
            effect_threshold: Nonnegative change magnitude below which a reference
                effect is treated as zero. Specify in the input measurement units.
            split_name: Label included in every output row.

        Returns:
            Tidy DataFrame with explicit counts, scope, reference source, direction
            of improvement, and status. Unavailable or undefined scores are NaN,
            never perfect scores. ``attrs['pertpy_evaluation']`` records the
            measurement choice, features, configuration and membership hashes.
            No aggregate silently drops unavailable groups. E-distance uses
            Pertpy's within-group estimator and can be negative in finite samples;
            a point baseline does not model the cell distribution.
        """
        allowed_metrics = {"mse", "edistance", "delta_pearson", "direction_accuracy", "top_k_overlap"}
        if len(metrics) == 0 or len(set(metrics)) != len(metrics) or set(metrics) - allowed_metrics:
            raise ValueError("Provide distinct supported metrics.")
        if len(set(baselines)) != len(baselines) or set(baselines) - {"control_mean", "additive"}:
            raise ValueError("Provide distinct supported baselines.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not np.isfinite(effect_threshold) or effect_threshold < 0:
            raise ValueError("effect_threshold must be finite and nonnegative.")
        if not predictions and not baselines:
            raise ValueError("Provide at least one prediction or baseline.")
        if any(not isinstance(name, str) or not name or name.startswith("baseline:") for name in predictions):
            raise ValueError("Model names must be nonempty strings and cannot start with 'baseline:'.")
        real_groups, train_groups = self._groups(ground_truth), self._groups(train)
        if len(ground_truth.obs_names.intersection(train.obs_names)):
            raise ValueError("Training and test cell identifiers overlap; split the data before evaluation.")
        features = ground_truth.var_names
        real_matrix = self._matrix(ground_truth, features)
        train_matrix = self._matrix(train, features)
        target_groups = {key: rows for key, rows in real_groups.items() if key[-1] != self.control}
        if not target_groups:
            raise ValueError("Ground truth contains no perturbed test groups.")
        if feature_sets is not None and set(feature_sets) - set(target_groups):
            raise ValueError("Feature sets must use test-group tuples as keys.")
        models = {}
        for name, data in predictions.items():
            groups = self._groups(data)
            if any(key not in target_groups and key[-1] != self.control for key in groups):
                raise ValueError(f"Prediction {name!r} contains groups absent from ground truth.")
            models[name] = (self._matrix(data, features), groups)
        rows = []
        for group, positions in target_groups.items():
            observed = real_matrix[positions]
            observed = observed if isinstance(observed, np.ndarray) else observed.toarray()
            observed = np.asarray(observed, dtype=np.float64)
            control_group = (*group[:-1], self.control)
            reference, reference_source = None, "unavailable"
            if control_group in train_groups:
                reference = np.asarray(
                    train_matrix[train_groups[control_group]].astype(np.float64).mean(axis=0)
                ).ravel()
                reference_source = "train_control"
            scopes = {"all": np.arange(len(features))}
            if feature_sets is not None and group in feature_sets:
                genes = feature_sets[group]
                if isinstance(genes, str) or len(genes) == 0 or len(set(genes)) != len(genes):
                    raise ValueError("Feature sets must be nonempty sequences of distinct gene identifiers.")
                indices = features.get_indexer(pd.Index(genes))
                if (indices < 0).any():
                    raise ValueError("Selected genes are absent from the measurement feature space.")
                scopes["selected"] = indices
            candidates = {}
            for name, (matrix, groups) in models.items():
                candidates[name] = (matrix[groups[group]], "ok") if group in groups else (None, "missing_prediction")
            for baseline in baselines:
                candidates[f"baseline:{baseline}"] = self._baseline(
                    baseline, group, train_matrix, train_groups, components
                )
            for model, (candidate, status) in candidates.items():
                predicted = None
                if candidate is not None:
                    predicted = candidate if isinstance(candidate, np.ndarray) else candidate.toarray()
                    predicted = np.asarray(predicted, dtype=np.float64)
                for scope, indices in scopes.items():
                    values = (
                        self._scores(
                            observed[:, indices],
                            predicted[:, indices],
                            reference[indices] if reference is not None else None,
                            features.to_numpy()[indices],
                            metrics=metrics,
                            top_k=top_k,
                            effect_threshold=effect_threshold,
                        )
                        if predicted is not None
                        else dict.fromkeys(metrics, (np.nan, status))
                    )
                    for metric, (value, metric_status) in values.items():
                        rows.append(
                            {
                                "model": model,
                                "split": split_name,
                                "context": group[0] if self.context_key else None,
                                "perturbation": group[-1],
                                "scope": scope,
                                "metric": metric,
                                "value": value,
                                "status": metric_status,
                                "higher_is_better": metric not in ("mse", "edistance"),
                                "n_true": len(observed),
                                "n_predicted": len(predicted) if predicted is not None else 0,
                                "n_features": len(indices),
                                "reference_source": reference_source,
                            }
                        )
        result = pd.DataFrame(rows)
        result.attrs["pertpy_evaluation"] = {
            "schema_version": 1,
            "perturbation_key": self.perturbation_key,
            "control": self.control,
            "context_key": self.context_key,
            "layer_key": self.layer_key,
            "features": features.tolist(),
            "train_cell_ids_sha256": _digest(train.obs_names.tolist()),
            "test_cell_ids_sha256": _digest(ground_truth.obs_names.tolist()),
            "n_train": train.n_obs,
            "n_test": ground_truth.n_obs,
            "split": split_name,
            "metrics": list(metrics),
            "baselines": list(baselines),
            "top_k": top_k,
            "effect_threshold": effect_threshold,
            "components": {key: list(value) for key, value in (components or {}).items()},
            "feature_sets": [
                {"group": list(key), "features": list(value)} for key, value in (feature_sets or {}).items()
            ],
        }
        return result
