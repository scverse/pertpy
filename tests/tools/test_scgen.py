import pytest

try:
    import scvi
except Exception:  # noqa: BLE001
    pytest.skip("Required R package 'edgeR' not available", allow_module_level=True)

import warnings

import anndata as ad
import jax
import jax.numpy as jnp
import scanpy as sc
from scvi import REGISTRY_KEYS

import pertpy as pt
from pertpy.tools._scgen._base_components import FlaxEncoder
from pertpy.tools._scgen._scgenvae import JaxSCGENVAE


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
    eval_adata = ad.concat([ctrl_adata, stim_adata, pred], label="concat_batches", index_unique="-")
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


def test_scgen_reconstruction_loss_is_per_cell():
    """The reconstruction loss must be summed per cell, matching PyTorch scGen, not over the whole minibatch."""
    module = JaxSCGENVAE(n_input=5)
    reconstruction_loss = module.get_reconstruction_loss(jnp.zeros((4, 5)), jnp.ones((4, 5)))
    assert reconstruction_loss.shape == (4,)
    assert jnp.allclose(reconstruction_loss, 5.0)


def test_scgen_decoder_uses_batch_norm():
    """With ``use_batch_norm="both"`` the decoder must use batch norm, matching PyTorch scGen."""
    module = JaxSCGENVAE(n_input=6, n_hidden=8, n_latent=3, n_layers=2)
    rngs = {name: jax.random.PRNGKey(i) for i, name in enumerate(module.required_rngs)}
    variables = module.init(rngs, {REGISTRY_KEYS.X_KEY: jnp.ones((4, 6))})
    decoder_params = variables["params"]["decoder"]
    assert any(key.startswith("BatchNorm") for key in decoder_params)


def test_scgen_encoder_applies_var_eps():
    """The encoder must add ``var_eps`` to the latent variance, matching PyTorch scGen."""
    encoder = FlaxEncoder(n_latent=3, n_hidden=8, n_layers=1, dropout_rate=0.0, use_batch_norm=False, training=False)
    x = jnp.ones((2, 4))
    variables = encoder.init({"params": jax.random.PRNGKey(0)}, x)
    _, var = encoder.apply(variables, x)
    _, var_without_eps = encoder.clone(var_eps=0.0).apply(variables, x)
    assert jnp.allclose(var - var_without_eps, 1e-4)
