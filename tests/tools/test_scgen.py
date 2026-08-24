import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import pertpy as pt
from pertpy.tools._scgen._base_components import FlaxEncoder
from pertpy.tools._scgen._scgenvae import JaxSCGENVAE


@pytest.fixture
def adata(rng) -> ad.AnnData:
    n_obs, n_vars = 400, 50
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(rng.choice(["batch_0", "batch_1"], n_obs)),
            "labels": pd.Categorical(rng.choice(["label_0", "label_1"], n_obs)),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    adata = ad.AnnData(rng.normal(5.0, 1.0, size=(n_obs, n_vars)).astype(np.float32), obs=obs)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    return adata


@pytest.fixture
def trained_model(adata) -> pt.tl.Scgen:
    pt.tl.Scgen.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = pt.tl.Scgen(adata, n_hidden=32, n_latent=8, n_layers=2)
    model.train(max_epochs=2, batch_size=32, early_stopping=True, early_stopping_patience=25)
    return model


def test_scgen(adata, trained_model):
    trained_model.batch_removal()

    pred, _delta = trained_model.predict(ctrl_key="batch_0", stim_key="batch_1", celltype_to_predict="label_0")
    pred.obs["batch"] = "pred"

    ctrl_adata = adata[((adata.obs["labels"] == "label_0") & (adata.obs["batch"] == "batch_0"))]
    stim_adata = adata[((adata.obs["labels"] == "label_0") & (adata.obs["batch"] == "batch_1"))]
    eval_adata = ad.concat([ctrl_adata, stim_adata, pred], label="concat_batches", index_unique="-")
    label_0 = adata[adata.obs["labels"] == "label_0"]
    sc.tl.rank_genes_groups(label_0, groupby="batch", method="wilcoxon")
    diff_genes = label_0.uns["rank_genes_groups"]["names"]["batch_1"]

    trained_model.plot_reg_mean_plot(
        eval_adata,
        condition_key="batch",
        axis_keys={"x": "pred", "y": "batch_1"},
        gene_list=diff_genes[:10],
        labels={"x": "predicted", "y": "ground truth"},
        save=False,
        show=False,
        legend=False,
    )

    trained_model.plot_reg_var_plot(
        eval_adata,
        condition_key="batch",
        axis_keys={"x": "pred", "y": "batch_1"},
        gene_list=diff_genes[:10],
        labels={"x": "predicted", "y": "ground truth"},
        save=False,
        show=False,
        legend=False,
    )


def test_scgen_does_not_require_scvi_or_torch():
    """scGen is a pure JAX implementation; importing it must not pull in a second runtime."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pertpy as pt; pt.tl.Scgen; "
            "print([m for m in ('scvi', 'torch', 'lightning') if m in sys.modules])",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_scgen_train_records_history(trained_model):
    assert len(trained_model.history["train_loss"]) == 2
    assert len(trained_model.history["validation_loss"]) == 2
    assert trained_model.is_trained


def test_scgen_latent_and_decoded_shapes(adata, trained_model):
    assert trained_model.get_latent_representation().shape == (adata.n_obs, 8)
    assert trained_model.get_decoded_expression().shape == (adata.n_obs, adata.n_vars)


def test_scgen_save_load_roundtrip(tmp_path, trained_model):
    latent = trained_model.get_latent_representation()

    trained_model.save(tmp_path / "model", save_anndata=True)
    loaded = pt.tl.Scgen.load(tmp_path / "model")

    assert loaded.is_trained
    assert loaded.history == trained_model.history
    np.testing.assert_allclose(loaded.get_latent_representation(), latent)


def test_scgen_save_refuses_to_overwrite(tmp_path, trained_model):
    trained_model.save(tmp_path / "model")
    with pytest.raises(FileExistsError):
        trained_model.save(tmp_path / "model")
    trained_model.save(tmp_path / "model", overwrite=True)


def test_scgen_load_without_anndata_is_an_error(tmp_path, trained_model):
    trained_model.save(tmp_path / "model")
    with pytest.raises(ValueError, match="saved without its AnnData"):
        pt.tl.Scgen.load(tmp_path / "model")


def test_scgen_requires_setup_anndata(adata):
    with pytest.raises(ValueError, match="setup_anndata"):
        pt.tl.Scgen(adata)


def test_scgen_setup_anndata_rejects_unknown_keys(adata):
    with pytest.raises(KeyError, match="batch_key"):
        pt.tl.Scgen.setup_anndata(adata, batch_key="does_not_exist", labels_key="labels")


def test_scgen_setup_anndata_defaults_to_constant_columns(adata):
    pt.tl.Scgen.setup_anndata(adata)
    model = pt.tl.Scgen(adata, n_hidden=8, n_latent=2)
    assert adata.obs[model.batch_key].nunique() == 1
    assert adata.obs[model.labels_key].nunique() == 1


def test_scgen_rejects_mismatched_var_names(adata, trained_model):
    other = adata[:, :10].copy()
    with pytest.raises(ValueError, match="var_names"):
        trained_model.get_latent_representation(other)


def test_scgen_untrained_model_raises(adata):
    pt.tl.Scgen.setup_anndata(adata, batch_key="batch", labels_key="labels")
    model = pt.tl.Scgen(adata, n_hidden=8, n_latent=2)
    with pytest.raises(RuntimeError, match="train the model first"):
        model.get_latent_representation()


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
    variables = module.init(rngs, jnp.ones((4, 6)))
    decoder_params = variables["params"]["decoder"]
    assert any(key.startswith("BatchNorm") for key in decoder_params)


def test_scgen_encoder_applies_var_eps():
    """The encoder must add ``var_eps`` to the latent variance, matching PyTorch scGen."""
    encoder = FlaxEncoder(n_latent=3, n_hidden=8, n_layers=1, dropout_rate=0.0, use_batch_norm=False, training=False)
    x = jnp.ones((2, 4))
    variables = encoder.init({"params": jax.random.PRNGKey(0)}, x)
    _, var = encoder.apply(variables, x)
    _, var_without_eps = encoder.clone(var_eps=0.0).apply(variables, x)
    # The encoder runs in float32, so compare against var_eps at float32 precision.
    assert jnp.allclose(var - var_without_eps, 1e-4, rtol=1e-2)
