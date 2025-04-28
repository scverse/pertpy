from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import issparse
from scipy.sparse import vstack as sp_vstack
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PerturbationComparison:
    """Comparison between real and simulated perturbations."""

    def compare_classification(
        self,
        real: np.ndarray,
        simulated: np.ndarray,
        control: np.ndarray,
        clf: ClassifierMixin | None = None,
    ) -> float:
        """Compare classification accuracy between real and simulated perturbations.

        Trains a classifier on the real perturbation data + the control data and reports a normalized
        classification accuracy on the simulated perturbation.

        Args:
            real: Real perturbed data.
            simulated: Simulated perturbed data.
            control: Control data
            clf: sklearn classifier to use, `sklearn.linear_model.LogisticRegression` if not provided.
        """
        assert real.shape[1] == simulated.shape[1] == control.shape[1]
        if clf is None:
            clf = LogisticRegression()
        n_x = real.shape[0]
        data = sp_vstack((real, control)) if issparse(real) else np.vstack((real, control))
        labels = np.concatenate([np.full(real.shape[0], "comp"), np.full(control.shape[0], "ctrl")])

        clf.fit(data, labels)
        norm_score = clf.score(simulated, np.full(simulated.shape[0], "comp")) / clf.score(real, labels[:n_x])
        norm_score = min(1.0, norm_score)

        return norm_score

    def compare_knn(
        self,
        real: np.ndarray,
        simulated: np.ndarray,
        control: np.ndarray | None = None,
        use_simulated_for_knn: bool = False,
        n_neighbors: int = 20,
        random_state: int = 0,
        n_jobs: int = 1,
    ) -> dict[str, float]:
        """Calculate proportions of real perturbed and control data points for simulated data.

        Computes proportions of real perturbed, control and simulated (if `use_simulated_for_knn=True`)
        data points for simulated data. If control (`C`) is not provided, builds the knn graph from
        real perturbed + simulated perturbed.

        Args:
            real: Real perturbed data.
            simulated: Simulated perturbed data.
            control: Control data
            use_simulated_for_knn: Include simulted perturbed data (`simulated`) into the knn graph. Only valid when
                control (`control`) is provided.
            n_neighbors: Number of neighbors to use in k-neighbor graph.
            random_state: Random state used for k-neighbor graph construction.
            n_jobs: Number of cores to use. Defaults to -1 (all).

        """
        assert real.shape[1] == simulated.shape[1]
        if control is not None:
            assert real.shape[1] == control.shape[1]

        n_y = simulated.shape[0]

        if control is None:
            index_data = sp_vstack((simulated, real)) if issparse(real) else np.vstack((simulated, real))
        else:
            datas = (simulated, real, control) if use_simulated_for_knn else (real, control)
            index_data = sp_vstack(datas) if issparse(real) else np.vstack(datas)

        y_in_index = use_simulated_for_knn or control is None
        c_in_index = control is not None
        label_groups = ["comp"]
        labels: NDArray[np.str_] = np.full(index_data.shape[0], "comp")
        if y_in_index:
            labels[:n_y] = "siml"
            label_groups.append("siml")
        if c_in_index:
            labels[-control.shape[0] :] = "ctrl"
            label_groups.append("ctrl")

        from pynndescent import NNDescent

        index = NNDescent(
            index_data,
            n_neighbors=max(50, n_neighbors),
            random_state=random_state,
            n_jobs=n_jobs,
        )
        indices = index.query(simulated, k=n_neighbors)[0]

        uq, uq_counts = np.unique(labels[indices], return_counts=True)
        uq_counts_norm = uq_counts / uq_counts.sum()
        counts = dict(zip(label_groups, [0.0] * len(label_groups), strict=False))
        counts = dict(zip(uq, uq_counts_norm, strict=False))

        return counts
