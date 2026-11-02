import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pertpy as pt
from pertpy.tools._perturbation_efficacy._mixscape import MixscapeGaussianMixture, _mixscape_gmm_predict_proba

NUM_CELLS_PER_GROUP = 10
ACCURACY_THRESHOLD = 0.8


def test_mixscape(adata):
    adata.layers["X_pert"] = adata.X
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.mixscape(adata=adata, pert_key="gene_target", control="NT", test_method="t-test")

    np_result = adata.obs["mixscape_class_global"] == "NP"
    np_result_correct = np_result[NUM_CELLS_PER_GROUP : NUM_CELLS_PER_GROUP * 2]
    ko_result = adata.obs["mixscape_class_global"] == "KO"
    ko_result_correct = ko_result[NUM_CELLS_PER_GROUP * 2 : NUM_CELLS_PER_GROUP * 3]

    assert "mixscape_class" in adata.obs
    assert "mixscape_class_global" in adata.obs
    assert "mixscape_class_p_ko" in adata.obs
    assert sum(np_result_correct) > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP
    assert sum(ko_result_correct) > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP


def _barplot_adata(n_genes: int, *, control: str = "NT", perturbation_type: str = "KO") -> ad.AnnData:
    guides: list[str] = []
    classes: list[str] = []
    for i in range(n_genes):
        guide = f"GENE{i}g1"
        guides.extend([guide, guide])
        classes.extend([perturbation_type, "NP"])
    guides.extend([f"{control}g1", f"{control}g1"])
    classes.extend([control, control])
    return ad.AnnData(
        X=np.zeros((len(guides), 1)),
        obs=pd.DataFrame({"guide": guides, "mixscape_class_global": classes}),
    )


def test_mixscape_barplot_handles_partial_subplot_rows():
    adata = _barplot_adata(6)
    fig = pt.tl.Mixscape().plot_barplot(adata, "guide", return_fig=True)

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(fig.axes) == 10
    assert len(visible_axes) == 6
    assert {ax.get_title() for ax in visible_axes} == {f"GENE{i}" for i in range(6)}
    plt.close(fig)


def test_mixscape_barplot_respects_custom_class_labels():
    adata = _barplot_adata(2, control="CTRL", perturbation_type="KD")
    fig = pt.tl.Mixscape().plot_barplot(
        adata,
        "guide",
        control="CTRL",
        perturbation_type="KD",
        return_fig=True,
    )

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 2
    assert {ax.get_title() for ax in visible_axes} == {"GENE0", "GENE1"}
    plt.close(fig)


def test_mixscape_min_cells_is_independent_of_min_de_genes():
    rng = np.random.default_rng(7)
    n_cells = 6
    n_genes = 20
    control = rng.normal(1.0, 0.1, size=(n_cells, n_genes))
    perturbed = rng.normal(5.0, 0.1, size=(n_cells, n_genes))
    adata = ad.AnnData(
        X=np.vstack([control, perturbed]),
        obs=pd.DataFrame(
            {"gene_target": ["NT"] * n_cells + ["target_gene"] * n_cells},
            index=[f"cell_{i}" for i in range(2 * n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    adata.layers["X_pert"] = adata.X.copy()

    result = pt.tl.Mixscape().mixscape(
        adata,
        pert_key="gene_target",
        control="NT",
        test_method="t-test",
        min_de_genes=10,
        min_cells=3,
        copy=True,
    )
    target = result.obs["gene_target"] == "target_gene"
    assert (result.obs.loc[target, "mixscape_class_global"] == "KO").all()

    result = pt.tl.Mixscape().mixscape(
        adata,
        pert_key="gene_target",
        control="NT",
        test_method="t-test",
        min_de_genes=1,
        min_cells=7,
        copy=True,
    )
    assert (result.obs.loc[target, "mixscape_class_global"] == "NP").all()


def test_mixscape_sparse_without_scaling_matches_dense(adata):
    adata.layers["X_pert"] = adata.X.toarray()
    dense = pt.tl.Mixscape().mixscape(
        adata=adata, pert_key="gene_target", control="NT", test_method="t-test", scale=False, copy=True
    )
    adata.layers["X_pert"] = adata.X
    pt.tl.Mixscape().mixscape(adata=adata, pert_key="gene_target", control="NT", test_method="t-test", scale=False)

    pd.testing.assert_frame_equal(adata.obs, dense.obs)


def test_perturbation_signature(adata):
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(adata, pert_key="label", control="control")

    assert "X_pert" in adata.layers


def test_lda(adata):
    adata.layers["X_pert"] = adata.X
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.mixscape(adata=adata, pert_key="gene_target", control="NT", test_method="t-test")
    mixscape_identifier.lda(adata=adata, pert_key="gene_target", control="NT", test_method="t-test")

    assert "mixscape_lda" in adata.uns


def test_deterministic_perturbation_signature():
    n_genes = 5
    n_cells_per_class = 50
    cell_classes = ["NT", "KO", "NP"]
    groups = ["Group1", "Group2"]

    cell_classes_array = np.repeat(cell_classes, n_cells_per_class)
    groups_array = np.tile(np.repeat(groups, n_cells_per_class // 2), len(cell_classes))
    obs = pd.DataFrame(
        {
            "cell_class": cell_classes_array,
            "group": groups_array,
            "perturbation": ["control" if cell_class == "NT" else "pert1" for cell_class in cell_classes_array],
        }
    )

    data = np.zeros((len(obs), n_genes))
    pert_effect = np.random.default_rng().uniform(-1, 1, size=(n_cells_per_class // len(groups), n_genes))
    for group in groups:
        baseline_expr = 2 if group == "Group1" else 10
        group_mask = obs["group"] == group
        data[(obs["cell_class"] == "NT") & group_mask] = baseline_expr
        data[(obs["cell_class"] == "KO") & group_mask] = baseline_expr + pert_effect
        data[(obs["cell_class"] == "NP") & group_mask] = baseline_expr

    var = pd.DataFrame(index=[f"Gene{i + 1}" for i in range(n_genes)])
    adata = ad.AnnData(X=data, obs=obs, var=var)

    for ref_selection_mode in ("nn", "split_by"):
        adata.layers.pop("X_pert", None)
        pt.tl.Mixscape().perturbation_signature(
            adata,
            pert_key="perturbation",
            control="control",
            ref_selection_mode=ref_selection_mode,
            n_neighbors=5,
            split_by="group",
        )
        assert "X_pert" in adata.layers
        assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NT"], 0)
        assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NP"], 0)
        assert np.allclose(
            adata.layers["X_pert"][obs["cell_class"] == "KO"], -np.concatenate([pert_effect] * len(groups), axis=0)
        )


def test_mixscape_gaussian_mixture():
    X = np.random.default_rng().random(100)
    fixed_means = [0.2, None]
    fixed_covariances = [None, 0.1]

    model = MixscapeGaussianMixture(n_components=2, fixed_means=fixed_means, fixed_covariances=fixed_covariances)
    model.fit(X.reshape(-1, 1))

    assert np.allclose(model.means_[0], fixed_means[0])
    assert np.allclose(model.covariances_[1], fixed_covariances[1])
    assert np.allclose(model.precisions_[1], 1 / fixed_covariances[1])


def test_mixscape_gmm_predict_proba_matches_sklearn():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(2.5, 1, 100)])
    guide_idx = np.arange(300, 400)
    means_init = np.array([x[:300].mean(), x[300:].mean()])
    precisions_init = 1 / np.array([x[:300].std(ddof=1), x[300:].std(ddof=1)]) ** 2
    expected = (
        MixscapeGaussianMixture(
            n_components=2,
            covariance_type="spherical",
            means_init=means_init[:, None],
            precisions_init=precisions_init,
            random_state=0,
            max_iter=100,
            fixed_means=[means_init[0], None],
            fixed_covariances=[1 / precisions_init[0], None],
        )
        .fit(x[:, None])
        .predict_proba(x[guide_idx, None])
    )

    proba = _mixscape_gmm_predict_proba(
        x, guide_idx, means_init=means_init, precisions_init=precisions_init, random_state=0
    )

    np.testing.assert_allclose(proba, expected, rtol=1e-9, atol=1e-12)
