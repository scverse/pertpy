"""Read and load input data into anndata object."""
from __future__ import annotations

import pandas as pd
from anndata import AnnData
from pandas import DataFrame
from rich import print
from sklearn.preprocessing import LabelEncoder


def load(
    input: AnnData | DataFrame,
    meta: DataFrame | None = None,
    label_col: str = "label_col",
    cell_type_col: str = "cell_type_col",
    condition_label: str | None = None,
    treatment_label: str | None = None,
) -> AnnData:
    """Loads the input data.

    Args:
        input: Anndata or matrix containing gene expression values (genes in rows, cells in columns) and optionally meta
            data about each cell.
        meta: Optional Pandas DataFrame containing meta data about each cell.
        label_col: column of the meta DataFrame or the Anndata or matrix containing the condition labels for each cell
            in the cell-by-gene expression matrix
        cell_type_col: column of the meta DataFrame or the Anndata or matrix containing the cell type labels for each
            cell in the cell-by-gene expression matrix
        condition_label: in the case of more than two labels, this label is used in the analysis
        treatment_label: in the case of more than two labels, this label is used in the analysis

    Returns:
        Anndata object containing gene expression values (cells in rows, genes in columns) and cell type, label and y
        dummie variables as obs
    """
    if isinstance(input, AnnData):
        input.obs = input.obs.rename(columns={cell_type_col: "cell_type", label_col: "label"})
        adata = input

    elif isinstance(input, DataFrame):
        if meta is None:
            try:
                _ = input[cell_type_col]
                _ = input[label_col]
            except KeyError:
                print("[bold red]No column names matching cell_type_col and label_col.")

        label = input[label_col] if meta is None else meta[label_col]
        cell_type = input[cell_type_col] if meta is None else meta[cell_type_col]
        x = input.drop([label_col, cell_type_col], axis=1) if meta is None else input
        adata = AnnData(X=x, obs=pd.DataFrame({"cell_type": cell_type, "label": label}))

    if len(adata.obs["label"].unique()) < 2:
        raise ValueError("Less than two unique labels in dataset. At least two are needed for the analysis.")
    # dummie variables for categorical data
    if adata.obs["label"].dtype.name == "category":
        # filter samples acording to label
        if condition_label is not None and treatment_label is not None:
            print(f"Filtering samples with {condition_label} and {treatment_label} labels.")
            adata = AnnData.concatenate(
                adata[adata.obs["label"] == condition_label], adata[adata.obs["label"] == treatment_label]
            )
        label_encoder = LabelEncoder()
        adata.obs["y_"] = label_encoder.fit_transform(adata.obs["label"])
    else:
        y = adata.obs["label"].to_frame()
        y = y.rename(columns={"label": "y_"})
        adata.obs = pd.concat([adata.obs, y], axis=1)

    return adata
