import pytest

try:
    import scvi
except Exception:  # noqa: BLE001
    pytest.skip("Required R package 'edgeR' not available", allow_module_level=True)

import warnings

import anndata as ad
import pertpy as pt
import scanpy as sc


def test_scgen():
    from scvi.data import synthetic_iid

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation names are not unique")
        adata = synthetic_iid()
        adata.obs_names_make_unique()
    pt.tl.Scgen.setup_anndata(
        adata,
        batch_key="batch",
        labels_key="labels",
    )

    scg = pt.tl.Scgen(adata)
    scg.train(max_epochs=1, batch_size=32, early_stopping=True, early_stopping_patience=25)

    scg.batch_removal()

    # predict
    pred, delta = scg.predict(ctrl_key="batch_0", stim_key="batch_1", celltype_to_predict="label_0")
    pred.obs["batch"] = "pred"

    # reg mean and reg var
    ctrl_adata = adata[((adata.obs["labels"] == "label_0") & (adata.obs["batch"] == "batch_0"))]
    stim_adata = adata[((adata.obs["labels"] == "label_0") & (adata.obs["batch"] == "batch_1"))]
    eval_adata = ad.concat([ctrl_adata, stim_adata, pred], label="concat_batches")
    label_0 = adata[adata.obs["labels"] == "label_0"]
    sc.tl.rank_genes_groups(label_0, groupby="batch", method="wilcoxon")
    diff_genes = label_0.uns["rank_genes_groups"]["names"]["batch_1"]

    scg.plot_reg_mean_plot(
        eval_adata,
        condition_key="batch",
        axis_keys={"x": "pred", "y": "batch_1"},
        gene_list=diff_genes[:10],
        labels={"x": "predicted", "y": "ground truth"},
        save=False,
        show=False,
        legend=False,
    )

    scg.plot_reg_var_plot(
        eval_adata,
        condition_key="batch",
        axis_keys={"x": "pred", "y": "batch_1"},
        gene_list=diff_genes[:10],
        labels={"x": "predicted", "y": "ground truth"},
        save=False,
        show=False,
        legend=False,
    )
