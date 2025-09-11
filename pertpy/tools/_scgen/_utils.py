from __future__ import annotations

import numpy as np


def extractor(
    data,
    cell_type,
    condition_key,
    cell_type_key,
    ctrl_key,
    stim_key,
):
    """Returns a list of `data` files while filtering for a specific `cell_type`.

    Args:
        data: `~anndata.AnnData` Annotated data matrix
        cell_type: Specific cell type to be extracted from `data`.
        condition_key: Key for `.obs` of `data` where conditions can be found.
        cell_type_key: Key for `.obs` of `data` where cell types can be found.
        ctrl_key: Key for `control` part of the `data` found in `condition_key`.
        stim_key: Key for `stimulated` part of the `data` found in `condition_key`.

    Returns:
        List of `data` files while filtering for a specific `cell_type`.

    Example:
        .. code-block:: python

            import Scgen
            import anndata

            train_data = anndata.read("./data/train.h5ad")
            test_data = anndata.read("./data/test.h5ad")
            train_data_extracted_list = extractor(
                train_data, "CD4T", "conditions", "cell_type", "control", "stimulated"
            )
    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condition_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == ctrl_key)]
    condition_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == stim_key)]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == stim_key))]

    return [training, condition_1, condition_2, cell_with_both_condition]


def balancer(
    adata,
    cell_type_key,
):
    """Makes cell type populations equal.

    Args:
        adata: `~anndata.AnnData` Annotated data matrix.
        cell_type_key: key for `.obs` of `data` where cell types can be found.

    Returns:
        Equal cell type population Annotated data matrix.

    Example:
        .. code-block:: python

            import Scgen
            import anndata

            train_data = anndata.read("./train_kang.h5ad")
            train_ctrl = train_data[train_data.obs["condition"] == "control", :]
            train_ctrl = balancer(train_ctrl, "conditions", "cell_type")
    """
    class_names = np.unique(adata.obs[cell_type_key])
    class_pop = {}
    for cls in class_names:
        class_pop[cls] = adata[adata.obs[cell_type_key] == cls].shape[0]
    max_number = np.max(list(class_pop.values()))
    index_all = []
    for cls in class_names:
        class_index = np.array(adata.obs[cell_type_key] == cls)
        index_cls = np.nonzero(class_index)[0]
        rng = np.random.default_rng()
        index_cls_r = index_cls[rng.choice(len(index_cls), max_number)]
        index_all.append(index_cls_r)

    balanced_data = adata[np.concatenate(index_all)].copy()

    return balanced_data
