from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pertpy._types import cast_frame, cast_matrix
from pertpy.tools._perturbation_space._perturbation_space import (
    PerturbationSpace,
    _carry_constant_obs,
    _sklearn_random_state,
)

if TYPE_CHECKING:
    from pertpy._types import RandomStateLike


class LRClassifierSpace(PerturbationSpace):
    """Fits a logistic regression model to the data and takes the feature space as embedding.

    We fit one logistic regression model per perturbation. After training, the coefficients of the logistic regression
    model are used as the feature space. This results in one embedding per perturbation.
    """

    def compute(
        self,
        adata: AnnData,
        *,
        target_col: str = "perturbation",
        layer_key: str | None = None,
        embedding_key: str | None = None,
        test_split_size: float = 0.2,
        max_iter: int = 1000,
        random_state: RandomStateLike = 0,
    ) -> AnnData:
        """Fits a logistic regression model to the data and takes the coefficients of the logistic regression model as perturbation embedding.

        Args:
            adata: AnnData object of size cells x genes
            target_col: `.obs` column that stores the label of the perturbation applied to each cell.
            layer_key: Layer in adata to use.
            embedding_key: Key of the embedding in obsm to be used as data for the logistic regression classifier.
                Can only be specified if layer_key is None.
            test_split_size: Fraction of data to put in the test set.
            max_iter: Maximum number of iterations taken for the solvers to converge.
            random_state: Random seed for the train/test split and the classifier.

        Returns:
            AnnData object with one observation per perturbation, the logistic regression coefficients in `.X` and the perturbation labels in `.obs[target_col]`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.norman_2019()
            >>> rcs = pt.tl.LRClassifierSpace()
            >>> pert_embeddings = rcs.compute(adata, embedding_key="X_pca", target_col="perturbation_name")
        """
        if layer_key is not None and layer_key not in adata.layers:
            raise ValueError(f"Layer key {layer_key} not found in adata.")

        if embedding_key is not None and embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key {embedding_key} not found in adata.obsm.")

        if layer_key is not None and embedding_key is not None:
            raise ValueError("Cannot specify both layer_key and embedding_key.")

        if target_col not in adata.obs:
            raise ValueError(f"Column {target_col!r} does not exist in the .obs attribute.")

        if layer_key is not None:
            regression_data = adata.layers[layer_key]
        elif embedding_key is not None:
            regression_data = adata.obsm[embedding_key]
        else:
            regression_data = cast_matrix(adata.X)

        regression_labels = adata.obs[target_col]
        random_state = _sklearn_random_state(random_state)
        regression_model = LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=random_state)

        perturbations = list(regression_labels.unique())
        embeddings = []
        scores = []
        for perturbation in perturbations:
            labels = np.where(regression_labels == perturbation, 1, 0)
            X_train, X_test, y_train, y_test = train_test_split(
                regression_data, labels, test_size=test_split_size, stratify=labels, random_state=random_state
            )
            regression_model.fit(X_train, y_train)
            embeddings.append(regression_model.coef_)
            scores.append(regression_model.score(X_test, y_test))

        pert_adata = AnnData(X=np.concatenate(embeddings, axis=0))
        pert_adata.obs_names = [str(perturbation) for perturbation in perturbations]
        pert_adata.obs[target_col] = pd.Categorical(perturbations)
        pert_adata.obs["classifier_score"] = scores

        _carry_constant_obs(pert_adata, cast_frame(adata.obs), target_col)
        pert_adata.obs[target_col] = cast_frame(pert_adata.obs)[target_col].astype("category")

        return pert_adata
