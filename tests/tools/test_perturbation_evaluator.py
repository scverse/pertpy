import json

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad
from scipy import sparse
from scipy.spatial.distance import cdist, pdist

import pertpy as pt


def cells(values, labels, prefix, *, contexts=None, genes=None):
    values = np.asarray(values, dtype=float)
    obs = pd.DataFrame({"condition": labels}, index=[f"{prefix}-{i}" for i in range(len(labels))])
    if contexts is not None:
        obs["cell_type"] = contexts
    return AnnData(values, obs=obs, var=pd.DataFrame(index=genes or [f"g{i}" for i in range(values.shape[1])]))


def value(scores, model, metric, scope="all", perturbation="A+B"):
    row = scores.query("model == @model and metric == @metric and scope == @scope and perturbation == @perturbation")
    assert len(row) == 1
    return row.iloc[0]


@pytest.fixture
def experiment():
    # The combination is exactly additive; control mean alone is insufficient.
    train = cells([[10, 20, 30], [10, 20, 30], [12, 20, 29], [10, 23, 31]], ["ctrl", "ctrl", "A", "B"], "train")
    truth = cells([[12, 23, 30], [12, 23, 30]], ["A+B", "A+B"], "test")
    return pt.tl.PerturbationEvaluator("condition", "ctrl"), train, truth


def test_additive_and_control_baselines(experiment):
    evaluator, train, truth = experiment
    scores = evaluator.evaluate(truth, {}, train=train, components={"A+B": ["A", "B"]}, top_k=2)
    assert value(scores, "baseline:additive", "mse").value == pytest.approx(0)
    assert value(scores, "baseline:control_mean", "mse").value == pytest.approx(13 / 3)
    assert value(scores, "baseline:additive", "edistance").value == pytest.approx(0)
    assert value(scores, "baseline:additive", "delta_pearson").value == pytest.approx(1)
    assert value(scores, "baseline:additive", "direction_accuracy").value == pytest.approx(1)
    assert value(scores, "baseline:additive", "top_k_overlap").value == pytest.approx(1)
    assert value(scores, "baseline:control_mean", "delta_pearson").status == "undefined_constant"
    assert value(scores, "baseline:control_mean", "top_k_overlap").status == "no_rankable_effect"
    assert scores.n_predicted.eq(1).all()
    assert scores.reference_source.eq("train_control").all()


def test_hand_calculated_metric_oracle():
    train = cells([[10, 100, 1000, 10000]], ["ctrl"], "train")
    truth = cells([[12, 99, 1000, 10003]], ["A+B"], "test")
    predicted = cells([[11, 102, 1000, 10003]], ["A+B"], "pred")
    evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl")
    scores = evaluator.evaluate(truth, {"model": predicted}, train=train, top_k=2, baselines=())
    assert value(scores, "model", "mse").value == pytest.approx(2.5)
    assert value(scores, "model", "delta_pearson").value == pytest.approx(3 / np.sqrt(50))
    assert value(scores, "model", "direction_accuracy").value == pytest.approx(2 / 3)
    assert value(scores, "model", "top_k_overlap").value == pytest.approx(0.5)
    assert value(scores, "model", "edistance").value == pytest.approx(2 * np.sqrt(10))


def test_energy_distance_matches_independent_pairwise_oracle():
    a = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]])
    b = np.array([[1.0, 2.0], [5.0, 2.0]])
    train = cells([[0, 0]], ["ctrl"], "train")
    truth = cells(a, ["A+B"] * 3, "test")
    predicted = cells(b, ["A+B"] * 2, "pred")
    expected = 2 * cdist(a, b).mean() - pdist(a).mean() - pdist(b).mean()
    scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
        truth, {"model": predicted}, train=train, baselines=(), metrics=("edistance",)
    )
    assert scores.value.iloc[0] == pytest.approx(expected)
    assert scores.n_true.iloc[0] == 3 and scores.n_predicted.iloc[0] == 2


def test_negative_energy_estimate_is_preserved():
    truth = cells([[0], [2]], ["A+B", "A+B"], "test")
    train = cells([[0]], ["ctrl"], "train")
    scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
        truth, {"identical": truth.copy()}, train=train, baselines=(), metrics=("edistance",)
    )
    assert scores.value.iloc[0] == pytest.approx(-2)
    assert scores.status.iloc[0] == "ok"


@pytest.mark.parametrize("matrix_type", [np.asarray, sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array])
def test_feature_permutation_sparse_and_no_mutation(experiment, matrix_type):
    evaluator, train, truth = experiment
    predicted = truth[:, [2, 0, 1]].copy()
    train = train[:, [1, 2, 0]].copy()
    original_truth = truth.X.copy()
    for adata in (train, truth, predicted):
        adata.X = matrix_type(adata.X)
    scores = evaluator.evaluate(
        truth, {"model": predicted}, train=train, metrics=("mse", "delta_pearson"), baselines=()
    )
    assert value(scores, "model", "mse").value == pytest.approx(0)
    assert value(scores, "model", "delta_pearson").value == pytest.approx(1)
    actual = truth.X.toarray() if sparse.issparse(truth.X) else truth.X
    np.testing.assert_array_equal(actual, original_truth)
    assert predicted.var_names.tolist() == ["g2", "g0", "g1"]


def test_explicit_layer_ignores_x(experiment):
    _, train, truth = experiment
    predicted = truth.copy()
    for adata in (train, truth, predicted):
        adata.layers["measurement"] = adata.X.copy()
        adata.X[:] = -1000
    predicted.X[:] = 9999
    scores = pt.tl.PerturbationEvaluator("condition", "ctrl", layer_key="measurement").evaluate(
        truth, {"model": predicted}, train=train, baselines=(), metrics=("mse",)
    )
    assert scores.value.iloc[0] == 0
    assert scores.attrs["pertpy_evaluation"]["layer_key"] == "measurement"


def test_missing_prediction_stays_in_scorecard(experiment):
    evaluator, train, truth = experiment
    more = cells([[1, 2, 3]], ["C"], "more")
    from anndata import concat

    both = concat([truth, more])
    scores = evaluator.evaluate(both, {"partial": truth.copy()}, train=train)
    missing = scores[(scores.model == "partial") & (scores.perturbation == "C")]
    assert len(missing) == 5
    assert missing.value.isna().all() and missing.status.eq("missing_prediction").all()
    assert missing.n_predicted.eq(0).all()


def test_no_effect_does_not_manufacture_a_score():
    train = cells([[1, 2, 3]], ["ctrl"], "train")
    truth = cells([[1, 2, 3]], ["A+B"], "test")
    scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
        truth, {}, train=train, baselines=("control_mean",)
    )
    assert value(scores, "baseline:control_mean", "mse").value == 0
    assert value(scores, "baseline:control_mean", "direction_accuracy").status == "no_reference_effect"
    assert value(scores, "baseline:control_mean", "top_k_overlap").status == "no_rankable_effect"


def test_context_holdout_does_not_fit_test_controls():
    adata = cells(
        [[0, 0], [1, 2], [100, 100], [101, 102]],
        ["ctrl", "A", "ctrl", "A"],
        "cell",
        contexts=["source", "source", "target", "target"],
    )
    evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl", context_key="cell_type")
    train, test = evaluator.split(adata, holdout=["target"], strategy="context")
    predictions = test[test.obs.condition == "A"].copy()
    scores = evaluator.evaluate(test, {"model": predictions}, train=train)
    assert scores[scores.model.str.startswith("baseline:")].status.eq("missing_training_control").all()
    model = scores[scores.model == "model"]
    assert model[model.metric == "mse"].value.iloc[0] == 0
    assert model[model.metric == "delta_pearson"].status.iloc[0] == "missing_reference_control"


def test_missing_additive_component_has_no_fallback(experiment):
    evaluator, train, truth = experiment
    train = train[train.obs.condition != "B"].copy()
    scores = evaluator.evaluate(truth, {}, train=train, components={"A+B": ["A", "B"]})
    unavailable = scores[scores.model == "baseline:additive"]
    assert unavailable.status.eq("missing_training_component").all()
    assert unavailable.value.isna().all()
    assert scores[scores.model == "baseline:control_mean"].status.ne("missing_training_component").all()


def test_group_split_and_combination_aliases():
    adata = cells([[0], [1], [2], [3], [3], [4]], ["ctrl", "A", "B", "A+B", "B+A", "C"], "cell")
    evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl")
    train, test = evaluator.split(
        adata, holdout=["A+B"], strategy="combination", components={"A+B": ["A", "B"], "B+A": ["B", "A"]}
    )
    assert set(test.obs.condition) == {"A+B", "B+A"}
    assert set(train.obs.condition) == {"ctrl", "A", "B", "C"}
    assert not len(train.obs_names.intersection(test.obs_names))
    assert train.uns["pertpy_evaluation_split"] == test.uns["pertpy_evaluation_split"]
    test.X[:] = 99
    assert adata.X.max() == 4
    # A perturbation-label split does not promise cold atomic-target exclusion.
    train, test = evaluator.split(adata, holdout=["A"])
    assert "A+B" in set(train.obs.condition)
    assert set(test.obs.condition) == {"A"}


def test_selected_genes_and_manifest(experiment):
    evaluator, train, truth = experiment
    selected = {("A+B",): truth.var_names[:2]}
    scores = evaluator.evaluate(
        truth, {}, train=train, feature_sets=selected, metrics=("mse",), baselines=("control_mean",)
    )
    assert value(scores, "baseline:control_mean", "mse", "selected").value == pytest.approx(6.5)
    manifest = json.loads(json.dumps(scores.attrs["pertpy_evaluation"]))
    assert manifest["feature_sets"] == [{"group": ["A+B"], "features": ["g0", "g1"]}]
    assert manifest["train_cell_ids_sha256"] != manifest["test_cell_ids_sha256"]


def test_heldout_values_cannot_change_baseline(experiment):
    evaluator, train, truth = experiment
    scores = evaluator.evaluate(truth, {}, train=train, baselines=("control_mean",), metrics=("mse",))
    changed = truth.copy()
    changed.X += 7
    changed_scores = evaluator.evaluate(changed, {}, train=train, baselines=("control_mean",), metrics=("mse",))
    assert scores.value.iloc[0] == pytest.approx((2**2 + 3**2) / 3)
    assert changed_scores.value.iloc[0] == pytest.approx((9**2 + 10**2 + 7**2) / 3)


@pytest.mark.parametrize(
    "fault",
    [
        "overlap",
        "duplicate_feature",
        "duplicate_cell",
        "missing_feature",
        "extra_feature",
        "nan",
        "inf",
        "missing_label",
        "missing_column",
        "missing_layer",
        "unknown_group",
    ],
)
def test_invalid_inputs_fail(experiment, fault):
    evaluator, train, truth = experiment
    predicted = truth.copy()
    if fault == "overlap":
        truth.obs_names = train.obs_names[:2]
    elif fault == "duplicate_feature":
        predicted.var_names = ["g0", "g0", "g2"]
    elif fault == "duplicate_cell":
        truth.obs_names = ["same", "same"]
    elif fault == "missing_feature":
        predicted = predicted[:, :2].copy()
    elif fault == "extra_feature":
        predicted.var_names = ["extra", "g1", "g2"]
    elif fault in ("nan", "inf"):
        predicted.X[0, 0] = float(fault)
    elif fault == "missing_label":
        predicted.obs.loc[predicted.obs_names[0], "condition"] = None
    elif fault == "missing_column":
        del predicted.obs["condition"]
    elif fault == "missing_layer":
        evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl", layer_key="absent")
    else:
        predicted.obs["condition"] = "not-in-test"
    with pytest.raises(ValueError):
        evaluator.evaluate(truth, {"model": predicted}, train=train)


@pytest.mark.parametrize(
    "components",
    [
        {"A+B": "A+B"},
        {"A+B": ["A", "A"]},
        {"A+B": ["ctrl", "A"]},
        {"A+B": ["A+B", "A"]},
        {"A+B": ["X", "A"], "X": ["B", "C"]},
    ],
)
def test_invalid_components(experiment, components):
    evaluator, train, truth = experiment
    with pytest.raises(ValueError):
        evaluator.evaluate(truth, {}, train=train, components=components)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"top_k": True},
        {"effect_threshold": -1},
        {"effect_threshold": np.inf},
        {"metrics": ()},
        {"metrics": ("made-up",)},
        {"baselines": ("made-up",)},
        {"feature_sets": {("A+B",): ["absent"]}},
    ],
)
def test_invalid_configuration(experiment, kwargs):
    evaluator, train, truth = experiment
    with pytest.raises(ValueError):
        evaluator.evaluate(truth, {}, train=train, **kwargs)


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_delta_pearson_preserves_scale_invariance(magnitude):
    train = cells([[0, 0]], ["ctrl"], "train")
    truth = cells([[magnitude, -magnitude]], ["A+B"], "test")
    prediction = cells([[1, -1]], ["A+B"], "prediction")
    scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
        truth, {"model": prediction}, train=train, baselines=(), metrics=("delta_pearson",)
    )
    actual = value(scores, "model", "delta_pearson")
    assert actual.status == "ok"
    assert actual.value == pytest.approx(1)


@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix])
def test_float32_control_reduction_matches_independent_float64(storage):
    values = np.array([[0.1, 1.3], [0.2, 2.7], [0.3, 3.1]], dtype=np.float32)
    train = cells(values, ["ctrl"] * 3, "train")
    train.X = storage(values)
    truth = cells([[0, 0]], ["A+B"], "test")
    scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
        truth, {}, train=train, baselines=("control_mean",), metrics=("mse",)
    )
    expected = np.mean(values.astype(np.float64).mean(axis=0) ** 2)
    np.testing.assert_allclose(value(scores, "baseline:control_mean", "mse").value, expected, rtol=1e-14)


@pytest.mark.parametrize("empty_axis", ["observations", "features"])
def test_empty_measurements_cannot_produce_a_scorecard(experiment, empty_axis):
    evaluator, train, truth = experiment
    empty = truth[:0].copy() if empty_axis == "observations" else truth[:, :0].copy()
    with pytest.raises(ValueError, match="observations and features"):
        evaluator.evaluate(empty, {}, train=train)


def test_context_cannot_reuse_the_intervention_column():
    with pytest.raises(ValueError, match="must be different"):
        pt.tl.PerturbationEvaluator("condition", "ctrl", context_key="condition")


def test_backed_predictions_require_explicit_materialization(experiment, tmp_path):
    evaluator, train, truth = experiment
    path = tmp_path / "predictions.h5ad"
    truth.write_h5ad(path)
    backed = read_h5ad(path, backed="r")
    try:
        with pytest.raises(TypeError, match="in-memory"):
            evaluator.evaluate(truth, {"model": backed}, train=train, baselines=(), metrics=("mse",))
    finally:
        backed.file.close()


def test_complex_measurements_are_not_silently_cast_to_real(experiment):
    evaluator, train, truth = experiment
    predicted = truth.copy()
    predicted.X = predicted.X.astype(complex) + 1j
    with pytest.raises(ValueError, match="real numeric"):
        evaluator.evaluate(truth, {"model": predicted}, train=train, baselines=(), metrics=("mse",))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"holdout": ["A"], "strategy": "random_cells"}, "split strategy"),
        ({"holdout": []}, "nonempty sequence"),
        ({"holdout": "A"}, "nonempty sequence"),
        ({"holdout": ["A", "A"]}, "distinct labels"),
        ({"holdout": ["absent"]}, "must occur"),
        ({"holdout": ["ctrl"]}, "control cannot"),
        ({"holdout": ["A"], "strategy": "context"}, "requires context_key"),
        ({"holdout": ["A"], "strategy": "combination"}, "at least two"),
    ],
)
def test_invalid_holdout_cannot_silently_change_the_split_unit(kwargs, message):
    adata = cells([[0], [1], [2]], ["ctrl", "A", "B"], "cell")
    evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl")
    original = adata.copy()
    with pytest.raises(ValueError, match=message):
        evaluator.split(adata, **kwargs)
    np.testing.assert_array_equal(adata.X, original.X)
    pd.testing.assert_frame_equal(adata.obs, original.obs)
    assert "pertpy_evaluation_split" not in adata.uns


def test_context_holdout_cannot_remove_every_training_cell():
    adata = cells([[0], [1]], ["ctrl", "A"], "cell", contexts=["only", "only"])
    evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl", context_key="cell_type")
    with pytest.raises(ValueError, match="partitions must be nonempty"):
        evaluator.split(adata, holdout=["only"], strategy="context")


@pytest.mark.parametrize("name", ["baseline:control_mean", "baseline:custom", "", 1])
def test_model_names_cannot_impersonate_a_baseline(experiment, name):
    evaluator, train, truth = experiment
    with pytest.raises(ValueError, match="Model names"):
        evaluator.evaluate(truth, {name: truth.copy()}, train=train, metrics=("mse",))


def test_evaluation_requires_a_candidate_and_perturbed_truth(experiment):
    evaluator, train, truth = experiment
    with pytest.raises(ValueError, match="prediction or baseline"):
        evaluator.evaluate(truth, {}, train=train, baselines=())
    controls_only = truth.copy()
    controls_only.obs["condition"] = "ctrl"
    with pytest.raises(ValueError, match="no perturbed test groups"):
        evaluator.evaluate(controls_only, {}, train=train)


@pytest.mark.parametrize(
    ("selected", "message"),
    [
        ({"A+B": ["g0"]}, "test-group tuples"),
        ({("absent",): ["g0"]}, "test-group tuples"),
        ({("A+B",): "g0"}, "sequences of distinct gene"),
        ({("A+B",): []}, "sequences of distinct gene"),
        ({("A+B",): ["g0", "g0"]}, "sequences of distinct gene"),
    ],
)
def test_invalid_selected_scope_cannot_change_gene_weighting(experiment, selected, message):
    evaluator, train, truth = experiment
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate(truth, {}, train=train, feature_sets=selected, metrics=("mse",))


def test_overflowed_additive_prediction_stays_unavailable():
    # Every input is finite, but adding two component effects exceeds float64.
    train = cells([[0, 0], [1e308, 0], [1e308, 0]], ["ctrl", "A", "B"], "train")
    truth = cells([[1, 0]], ["A+B"], "test")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
            truth, {}, train=train, components={"A+B": ["A", "B"]}, metrics=("mse",)
        )
    unavailable = value(scores, "baseline:additive", "mse")
    assert unavailable.status == "nonfinite_baseline"
    assert np.isnan(unavailable.value)
    assert unavailable.n_predicted == 0
    available = value(scores, "baseline:control_mean", "mse")
    assert available.status == "ok"
    assert available.value == pytest.approx(0.5)


def test_overflowed_delta_cannot_be_ranked_as_a_real_effect():
    train = cells([[-1e308, 0]], ["ctrl"], "train")
    truth = cells([[1e308, 1]], ["A+B"], "test")
    predicted = truth.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
            truth,
            {"model": predicted},
            train=train,
            baselines=(),
            metrics=("mse", "delta_pearson", "direction_accuracy", "top_k_overlap"),
        )
    assert value(scores, "model", "mse").value == 0
    undefined = scores[scores.metric != "mse"]
    assert undefined.status.eq("nonfinite_result").all()
    assert undefined.value.isna().all()


def test_overflowed_distance_does_not_hide_a_finite_correlation():
    train = cells([[0, 0]], ["ctrl"], "train")
    truth = cells([[1e308, -1e308]], ["A+B"], "test")
    predicted = cells([[-1e308, 1e308]], ["A+B"], "pred")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = pt.tl.PerturbationEvaluator("condition", "ctrl").evaluate(
            truth, {"model": predicted}, train=train, baselines=(), metrics=("mse", "delta_pearson")
        )
    unavailable = value(scores, "model", "mse")
    assert unavailable.status == "nonfinite_result"
    assert np.isnan(unavailable.value)
    correlation = value(scores, "model", "delta_pearson")
    assert correlation.status == "ok"
    assert correlation.value == pytest.approx(-1)
