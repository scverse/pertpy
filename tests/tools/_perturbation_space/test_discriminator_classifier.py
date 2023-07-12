import numpy as np
import pandas as pd
from anndata import AnnData

import pertpy as pt


def test_discriminator_classifier():
    X = np.random.rand(20, 5)
    obs = pd.DataFrame(
        {
            "perturbations": [
                "control",
                "target1",
                "target1",
                "target2",
                "target2",
                "target1",
                "target1",
                "target2",
                "target2",
                "target2",
                "control",
                "target1",
                "target1",
                "target2",
                "target2",
                "target1",
                "target1",
                "target2",
                "target2",
                "target2",
            ]
        }
    )
    adata = AnnData(X, obs=obs)

    # Compute the embeddings using the classifier
    ps = pt.tl.DiscriminatorClassifierSpace()
    classifier_ps = ps(adata)
    classifier_ps.train(max_epochs=1)
    pert_embeddings = classifier_ps.get_embeddings()

    # Compute the pseudobulk of the given embeddings
    ps2 = pt.tl.PseudobulkSpace()
    psadata = ps2(pert_embeddings, copy=True)  # noqa
