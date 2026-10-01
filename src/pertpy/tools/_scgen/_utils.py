from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd


def extractor(
    data,
    cell_type,
    *,
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
                train_data,
                "CD4T",
                condition_key="conditions",
                cell_type_key="cell_type",
                ctrl_key="control",
                stim_key="stimulated",
            )
    """
    cell_with_both_condition = data[data.obs[cell_type_key] == cell_type]
    condition_1 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == ctrl_key)]
    condition_2 = data[(data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == stim_key)]
    training = data[~((data.obs[cell_type_key] == cell_type) & (data.obs[condition_key] == stim_key))]

    return [training, condition_1, condition_2, cell_with_both_condition]


def balancer(labels: pd.Series) -> np.ndarray:
    """Upsamples every cell type to the size of the largest one.

    Args:
        labels: Cell type of every cell.

    Returns:
        Positional indices into ``labels`` with equally many cells per cell type.
    """
    labels_arr = np.asarray(labels)
    class_names, class_pop = np.unique(labels_arr, return_counts=True)
    max_number = class_pop.max()
    index_all = []
    for cls in class_names:
        index_cls = np.flatnonzero(labels_arr == cls)
        rng = np.random.default_rng()
        index_all.append(index_cls[rng.choice(len(index_cls), max_number)])

    return np.concatenate(index_all)


def _padded_batches(x: np.ndarray, batch_size: int, *, pad: bool = True) -> Iterator[tuple[np.ndarray, int]]:
    """Yields row batches of ``x`` together with their number of rows.

    With ``pad``, the last batch is zero-padded to the shape of the others.
    """
    batch_size = min(batch_size, x.shape[0])
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size]
        n = batch.shape[0]
        if pad and n < batch_size:
            batch = np.pad(batch, ((0, batch_size - n), (0, 0)))
        yield batch, n
