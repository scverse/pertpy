# Evaluating perturbation predictions

`pt.tl.PerturbationEvaluator` compares held-out measurements with named predictions
and simple training-only baselines. It returns one row per model, intervention,
context, feature scope and metric, retaining unavailable scores with an explicit
status. It does not fit external models, normalize measurements or discover DEGs.

## A complete synthetic example

```python
import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt

adata = ad.AnnData(
    X=np.array([[10., 20., 30.], [12., 20., 29.],
                [10., 23., 31.], [12., 23., 30.]]),
    obs=pd.DataFrame({"condition": ["ctrl", "A", "B", "A+B"]},
                     index=["control-cell", "a-cell", "b-cell", "ab-cell"]),
    var=pd.DataFrame(index=["gene-1", "gene-2", "gene-3"]),
)
components = {"A+B": ["A", "B"]}
evaluator = pt.tl.PerturbationEvaluator("condition", "ctrl")
train, test = evaluator.split(
    adata, holdout=["A+B"], strategy="combination", components=components
)
scores = evaluator.evaluate(test, {}, train=train, components=components)
print(scores[["model", "metric", "value", "status"]])
```

The additive baseline predicts `[12, 23, 30]` and has zero mean squared error.
The control-mean baseline has MSE `13/3`. Its predicted expression change is zero,
so its delta correlation and effect ranking are undefined, rather than perfect.
These are point predictions; neither represents cell-to-cell variability.

Supply external predictions as `{"my_model": predicted_adata}`. Predictions must
carry the same observation grouping columns and exactly the same gene identifiers
as training and truth. Gene order may differ: alignment uses `var_names`. Numbers
of predicted and measured cells may differ; individual cells are not paired.
Missing prediction groups produce rows with `status="missing_prediction"`.

## Declare the question before scoring

- **Perturbation holdout** removes complete, explicitly named labels. Holding out
  `A` does not remove `A+B`; this is not an atomic-target cold-start split.
- **Combination holdout** uses an explicit component mapping. Include all known
  aliases, such as `{"A+B": ["A", "B"], "B+A": ["B", "A"]}`, to hold them out
  together. Labels are never split on punctuation automatically.
- **Context holdout** removes an entire cell type or other declared context,
  including its controls. Baselines cannot use controls from that held-out context.
  Absolute model errors remain scoreable; missing reference-dependent scores are
  reported explicitly. Other contexts are never silently pooled.

The split helper records exact cell membership in
`uns["pertpy_evaluation_split"]`. Evaluation rejects overlapping training and test
cell identifiers. These checks cannot establish how an external model was trained
or prevent upstream preprocessing leakage. Fit feature selection, preprocessing
and hyperparameters on training data; reserve validation data for model selection.

All inputs must be on the same declared scale. Choose `layer_key="lognorm"` to use
that layer consistently. Otherwise the evaluator uses `X`, never an implicit PCA
representation. Additivity on log-normalized values is a modelling baseline, not
a claim that molecular effects are additive in that space. Predictions are not
clipped to zero.

## Baselines and metrics

The control baseline uses the training control mean in the same context. The
additive baseline adds training single-intervention mean changes to that control
mean. Missing singles or controls produce unavailable rows; there is no fallback.
Nearest-neighbour baselines and log-fold-change scoring are outside this first
API: they require additional contracts for query representations and expression
units respectively.

| Metric | Interpretation |
| --- | --- |
| `mse` | Existing `Distance("mse")`: mean squared error between gene means; lower is better. |
| `edistance` | Existing `Distance("edistance")` on cell vectors; lower is better. The within-group estimator can be negative in finite samples. It is not clamped. |
| `delta_pearson` | Gene-wise Pearson correlation of mean changes from training controls. Constant changes are undefined. |
| `direction_accuracy` | Fraction of true changes above `effect_threshold` with the correct predicted sign. |
| `top_k_overlap` | Intersection fraction of the top absolute mean changes. Gene identifiers resolve ties; this is not a statistical DEG test. |

`feature_sets={("A+B",): ["gene-1", "gene-2"]}` adds a `selected` scope alongside
`all`. With a context column, keys are `(context, perturbation)` tuples. Declare
where these genes came from, and distinguish outcome-derived descriptive scores
from a predeclared test. `top_k` is capped at the number of genes in each scope.

Every metric has its own status. Keep unavailable rows visible when reporting
coverage or aggregating results. `scores.attrs["pertpy_evaluation"]` records the
configuration, feature sets and cell-membership hashes; save it separately when
exporting CSV, which does not preserve DataFrame attributes. Neither cell-level
scores nor resampling cells establishes biological-replicate uncertainty.
