from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as ss
import sklearn.metrics
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn, sinkhorn_lr
from scanpy.plotting import _utils
from scipy.sparse import issparse
from seaborn import heatmap
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.pyplot import Figure
    from statsmodels.tools.typing import ArrayLike


class Cinemaot:
    """CINEMA-OT is a causal framework for perturbation effect analysis to identify individual treatment effects and synergy."""

    def __init__(self):
        pass

    def causaleffect(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        return_matching: bool = False,
        cf_rep: str = "cf",
        use_rep: str = "X_pca",
        batch_size: int | None = None,
        dim: int | None = 20,
        thres: float = 0.15,
        smoothness: float = 1e-4,
        rank: int = 200,
        eps: float = 1e-3,
        solver: str = "Sinkhorn",
        preweight_label: str | None = None,
        random_state: int | None = 0,
    ):
        """Calculate the confounding variation, optimal transport counterfactual pairs, and single-cell level treatment effects.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            return_matching: Whether to return the matching matrix in the returned de.obsm['ot'].
            cf_rep: the place to put the confounder embedding in the original adata.obsm.
            use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
            batch_size: Size of batch to calculate the optimal transport map.
            dim: Use the first dim components in use_rep.
                 If none, use a biwhitening procedure on the raw count matrix to derive a reasonable rank.
            thres: the threshold for the rank-dependence metric.
            smoothness: the coefficient determining the smooth level in entropic optimal transport problem.
            rank: Only used if the solver "LRSinkhorn" is used. Specifies the rank number of the transport map.
            eps: Tolerate error of the optimal transport.
            solver: Either "Sinkhorn" or "LRSinkhorn". The ott-jax solver used.
            preweight_label: The annotated label (e.g. cell type) that is used to assign weights for treated
                             and control cells to balance across the label. Helps overcome the differential abundance issue.
            random_state: The random seed for the shuffling.

        Returns:
            Returns an AnnData object that contains the single-cell level treatment effect as de.X and the
            corresponding low dimensional embedding in de.obsm['X_embedding'], and optional matching matrix
            stored in the de.obsm['ot']. Also puts the confounding variation in adata.obsm[cf_rep].

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> model = pt.tl.Cinemaot()
            >>> out_adata = model.causaleffect(
            >>>         adata, pert_key="perturbation", control="No stimulation", return_matching=True,
            >>>         thres=0.5, smoothness=1e-5, eps=1e-3, solver="Sinkhorn", preweight_label="cell_type0528")
        """
        available_solvers = ["Sinkhorn", "LRSinkhorn"]
        if solver not in available_solvers:
            raise ValueError(f"solver = {solver} is not one of the supported solvers: {available_solvers}")

        if dim is None:
            dim = self.get_dim(adata, use_rep=use_rep)

        transformer = FastICA(n_components=dim, random_state=0, whiten="arbitrary-variance")
        X_transformed = np.array(transformer.fit_transform(adata.obsm[use_rep][:, :dim]), dtype=np.float64)
        groupvec = (adata.obs[pert_key] == control * 1).values  # control
        xi = np.zeros(dim)
        j = 0
        for source_row in X_transformed.T:
            xi_obj = Xi(source_row, groupvec * 1, random_state=random_state)
            xi[j] = xi_obj.correlation
            j = j + 1

        cf = np.array(X_transformed[:, xi < thres], np.float64)
        cf1 = np.array(cf[adata.obs[pert_key] == control, :], np.float64)
        cf2 = np.array(cf[adata.obs[pert_key] != control, :], np.float64)
        if sum(xi < thres) == 1:
            sklearn.metrics.pairwise_distances(cf1.reshape(-1, 1), cf2.reshape(-1, 1))
        elif sum(xi < thres) == 0:
            raise ValueError("No confounder components identified. Please try a higher threshold.")
        else:
            sklearn.metrics.pairwise_distances(cf1, cf2)

        e = smoothness * sum(xi < thres)
        geom = PointCloud(cf1, cf2, epsilon=e, batch_size=batch_size)

        if preweight_label is None:
            ot_prob = linear_problem.LinearProblem(geom, a=None, b=None)

        else:
            # Implement a simple function here, taking adata.obs, output inverse prob weight.
            # For consistency, c is still the empirical distribution, while r is weighted.
            a = np.zeros(cf1.shape[0])
            b = np.zeros(cf2.shape[0])

            adata1 = adata[adata.obs[pert_key] == control]
            adata2 = adata[adata.obs[pert_key] != control]

            for label in adata1.obs[pert_key].unique():
                mask_label = adata1.obs[pert_key] == label
                for ct in adata1.obs[preweight_label].unique():
                    mask_ct = adata1.obs[preweight_label] == ct
                    a[mask_ct & mask_label] = np.sum(adata2.obs[preweight_label] == ct) / np.sum(mask_ct)
                a[mask_label] /= np.sum(a[mask_label])

            a = a / np.sum(a)
            b[:] = 1 / cf2.shape[0]
            ot_prob = linear_problem.LinearProblem(geom, a=a, b=b)

        if solver == "LRSinkhorn":
            if rank is None:
                rank = int(min(cf1.shape[0], cf2.shape[0]) / 2)
            _solver = jax.jit(sinkhorn_lr.LRSinkhorn(rank=rank, threshold=eps))
            ot_sink = _solver(ot_prob)
            embedding = (
                X_transformed[adata.obs[pert_key] != control, :]
                - ot_sink.apply(X_transformed[adata.obs[pert_key] == control, :].T).T
                / ot_sink.apply(np.ones_like(X_transformed[adata.obs[pert_key] == control, :].T)).T
            )

            X = adata.X.toarray() if issparse(adata.X) else adata.X
            te2 = (
                X[adata.obs[pert_key] != control, :]
                - ot_sink.apply(X[adata.obs[pert_key] == control, :].T).T
                / ot_sink.apply(np.ones_like(X[adata.obs[pert_key] == control, :].T)).T
            )
            if issparse(X):
                del X

            adata.obsm[cf_rep] = cf
            adata.obsm[cf_rep][adata.obs[pert_key] != control, :] = (
                ot_sink.apply(adata.obsm[cf_rep][adata.obs[pert_key] == control, :].T).T
                / ot_sink.apply(np.ones_like(adata.obsm[cf_rep][adata.obs[pert_key] == control, :].T)).T
            )

        else:
            _solver = jax.jit(sinkhorn.Sinkhorn(threshold=eps))
            ot_sink = _solver(ot_prob)
            ot_matrix = np.array(ot_sink.matrix.T, dtype=np.float64)
            embedding = X_transformed[adata.obs[pert_key] != control, :] - np.matmul(
                ot_matrix / np.sum(ot_matrix, axis=1)[:, None], X_transformed[adata.obs[pert_key] == control, :]
            )

            X = adata.X.toarray() if issparse(adata.X) else adata.X

            te2 = X[adata.obs[pert_key] != control, :] - np.matmul(
                ot_matrix / np.sum(ot_matrix, axis=1)[:, None], X[adata.obs[pert_key] == control, :]
            )
            if issparse(X):
                del X

            adata.obsm[cf_rep] = cf
            adata.obsm[cf_rep][adata.obs[pert_key] != control, :] = np.matmul(
                ot_matrix / np.sum(ot_matrix, axis=1)[:, None], adata.obsm[cf_rep][adata.obs[pert_key] == control, :]
            )

        TE = sc.AnnData(np.array(te2), obs=adata[adata.obs[pert_key] != control, :].obs.copy(), var=adata.var.copy())
        TE.obsm["X_embedding"] = embedding

        if return_matching:
            TE.obsm["ot"] = np.asarray(ot_sink.matrix.T)
            return TE
        else:
            return TE

    def causaleffect_weighted(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        return_matching: bool = False,
        cf_rep: str = "cf",
        use_rep: str = "X_pca",
        batch_size: int | None = None,
        k: int = 20,
        dim: int | None = 20,
        thres: float = 0.15,
        smoothness: float = 1e-4,
        rank: int = 200,
        eps: float = 1e-3,
        solver: str = "Sinkhorn",
        resolution: float = 1.0,
    ):
        """The resampling CINEMA-OT algorithm that allows tackling the differential abundance in an unsupervised manner.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            return_matching: Whether to return the matching matrix in the returned de.obsm['ot'].
            cf_rep: the place to put the confounder embedding in the original adata.obsm.
            use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
            batch_size: Size of batch to calculate the optimal transport map.
            k: the number of neighbors used in the k-NN matching phase.
            dim: Use the first dim components in use_rep.
                 If None, use a biwhitening procedure on the raw count matrix to derive a reasonable rank.
            thres: the threshold for the rank-dependence metric.
            smoothness: the coefficient determining the smooth level in entropic optimal transport problem.
            rank: Only used if the solver "LRSinkhorn" is used. Specifies the rank number of the transport map.
            eps: Tolerate error of the optimal transport.
            solver: Either "Sinkhorn" or "LRSinkhorn". The ott-jax solver used.
            resolution: the clustering resolution used in the sampling phase.

        Returns:
            Returns an anndata object that contains the single-cell level treatment effect as de.X and the
            corresponding low dimensional embedding in de.obsm['X_embedding'], and optional matching matrix
            stored in the de.obsm['ot']. Also puts the confounding variation in adata.obsm[cf_rep].

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> model = pt.tl.Cinemaot()
            >>> ad, de = model.causaleffect_weighted(
            >>>              adata, pert_key="perturbation", control="No stimulation", return_matching=True,
            >>>              thres=0.5, smoothness=1e-5, eps=1e-3, solver="Sinkhorn")
        """
        available_solvers = ["Sinkhorn", "LRSinkhorn"]
        assert solver in available_solvers, (
            f"solver = {solver} is not one of the supported solvers: {available_solvers}"
        )

        if dim is None:
            dim = self.get_dim(adata, use_rep=use_rep)

        adata.obs_names_make_unique()

        idx = self._get_weightidx(
            adata, pert_key=pert_key, control=control, k=k, use_rep=use_rep, resolution=resolution
        )
        adata_ = adata[idx].copy()
        TE = self.causaleffect(
            adata_,
            pert_key,
            control,
            return_matching=return_matching,
            cf_rep=cf_rep,
            use_rep=use_rep,
            batch_size=batch_size,
            dim=dim,
            thres=thres,
            smoothness=smoothness,
            rank=rank,
            eps=eps,
            solver=solver,
        )
        return adata_, TE

    def generate_pseudobulk(
        self,
        adata: AnnData,
        de: AnnData,
        pert_key: str,
        control: str,
        label_list: list,
        cf_rep: str = "cf",
        de_rep: str = "X_embedding",
        cf_resolution: float = 0.5,
        de_resolution: float = 0.5,
        use_raw: bool = True,
    ):
        """Generating pseudobulk anndata considering the differential response behaviors revealed by CINEMA-OT.

        Requires running Cinemaot.causaleffect() or Cinemaot.causaleffect_weighted() first.

        Args:
            adata: The annotated data object.
            de: The anndata output from Cinemaot.causaleffect() or Cinemaot.causaleffect_weighted().
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            label_list: Additional covariate labels used to segragate pseudobulk.
                        Should at least contain sample information (sample 1, sample 2,..., etc).
            cf_rep: the place to put the confounder embedding in the original adata.obsm.
            de_rep: Use the indicated representation in de.obsm.
            assign_cf: If a str is passed, a label in adata.obs instead of confounder Leiden label is used.
            cf_resolution: The leiden clustering resolution for the confounder.
            de_resolution: The leiden clustering resolution for the differential response.
            use_raw: If true, use adata.raw.X to aggregate the pseudobulk profiles. Otherwise use adata.X.

        Returns:
            Returns an anndata object that contains aggregated pseudobulk profiles and associated metadata.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> model = pt.tl.Cinemaot()
            >>> de = model.causaleffect(
            >>>         adata, pert_key="perturbation", control="No stimulation", return_matching=True, thres=0.5,
            >>>         smoothness=1e-5, eps=1e-3, solver="Sinkhorn", preweight_label="cell_type0528")
            >>> adata_pb = model.generate_pseudobulk(
            >>>         adata, de, pert_key="perturbation", control="No stimulation", label_list=None)
        """
        sc.pp.neighbors(de, use_rep=de_rep)
        sc.tl.leiden(de, resolution=de_resolution)
        if use_raw:
            if issparse(adata.raw.X):
                df = pd.DataFrame(adata.raw.X.toarray(), columns=adata.raw.var_names, index=adata.raw.obs_names)
            else:
                df = pd.DataFrame(adata.raw.X, columns=adata.raw.var_names, index=adata.raw.obs_names)
        elif issparse(adata.X):
            df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
        else:
            df = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)

        if label_list is None:
            label_list = ["ct"]
            sc.pp.neighbors(adata, use_rep=cf_rep)
            sc.tl.leiden(adata, resolution=cf_resolution)
            df["ct"] = adata.obs["leiden"].astype(str)
        df["ptb"] = "control"
        df.loc[adata.obs[pert_key] != control, "ptb"] = de.obs["leiden"].astype(str)
        label_list.append("ptb")
        df = df.groupby(label_list).sum()
        new_index = df.index.map(lambda x: "_".join(map(str, x)))
        df_ = df.set_index(new_index)
        adata_pb = sc.AnnData(df_)
        adata_pb.obs = pd.DataFrame(
            df.index.to_frame().values,
            index=adata_pb.obs_names,
            columns=df.index.to_frame().columns,
        )
        return adata_pb

    def get_dim(
        self,
        adata: AnnData,
        c: float = 0.5,
        use_rep: str = "X_pca",
    ):
        """Estimating the rank of the count matrix. Always use adata.raw.X. Make sure it is the raw count matrix.

        Args:
            adata: The annotated data object.
            c: the parameter regarding the quadratic variance distribution. c=0 means Poisson count matrices.
            use_rep: the embedding used to give a upper bound for the estimated rank.

        Returns:
            Returns the estimated dimension number.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> model = pt.tl.Cinemaot()
            >>> dim = model.get_dim(adata)
        """
        sk = SinkhornKnopp()
        data = adata.raw.X.toarray() if issparse(adata.raw.X) else adata.raw.X
        vm = (1e-3 + data + c * data * data) / (1 + c)
        sk.fit(vm)
        wm = np.dot(np.dot(np.sqrt(sk._D1), vm), np.sqrt(sk._D2))
        u, s, vt = np.linalg.svd(wm)
        dim = min(sum(s > (np.sqrt(data.shape[0]) + np.sqrt(data.shape[1]))), adata.obsm[use_rep].shape[1])
        return dim

    def _get_weightidx(
        self,
        adata: AnnData,
        *,
        pert_key: str,
        control: str,
        use_rep: str = "X_pca",
        k: int = 20,
        resolution: float = 1.0,
    ):
        """Generating the resampled indices that balances covariates across treatment conditions.

        Args:
            adata: The annotated data object.
            pert_key: Key of the perturbation col.
            control: Key of the control col.
            use_rep: the embedding used to give a upper bound for the estimated rank.
            k: the number of neighbors used in the k-NN matching phase.
            resolution: the clustering resolution used in the sampling phase.

        Returns:
            Returns the indices.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> model = pt.tl.Cinemaot()
            >>> idx = model._get_weightidx(adata, pert_key="perturbation", control="No stimulation")
        """
        adata_ = adata.copy()
        X_pca1 = adata_.obsm[use_rep][adata_.obs[pert_key] == control, :]
        X_pca2 = adata_.obsm[use_rep][adata_.obs[pert_key] != control, :]
        nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X_pca1)
        mixscape_pca = adata.obsm[use_rep].copy()
        mixscapematrix = nbrs.kneighbors_graph(X_pca2).toarray()
        mixscape_pca[adata_.obs[pert_key] != control, :] = (
            np.dot(mixscapematrix, mixscape_pca[adata_.obs[pert_key] == control, :]) / k
        )

        adata_.obsm["X_mpca"] = mixscape_pca
        sc.pp.neighbors(adata_, use_rep="X_mpca")
        sc.tl.leiden(adata_, resolution=resolution)

        j = 0

        ref_label = "noncontrol"
        expr_label = "control"

        adata_.obs["ct"] = ref_label
        adata_.obs.loc[adata_.obs[pert_key] == control, "ct"] = expr_label
        pert_key = "ct"
        z = np.zeros(adata_.shape[0]) + 1

        for i in adata_.obs["leiden"].cat.categories:
            if (
                adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].shape[0]
                >= adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].shape[0]
            ):
                z[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)] = (
                    adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].shape[0]
                    / adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].shape[0]
                )
                if j == 0:
                    idx = sc.pp.subsample(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)],
                        n_obs=adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].shape[0],
                        copy=True,
                    ).obs.index
                    idx = idx.append(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].obs.index
                    )
                    j = j + 1
                else:
                    idx_tmp = sc.pp.subsample(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)],
                        n_obs=adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].shape[0],
                        copy=True,
                    ).obs.index
                    idx_tmp = idx_tmp.append(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].obs.index
                    )
                    idx = idx.append(idx_tmp)
            else:
                z[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)] = (
                    adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].shape[0]
                    / adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)].shape[0]
                )
                if j == 0:
                    idx = sc.pp.subsample(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)],
                        n_obs=adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].shape[0],
                        copy=True,
                    ).obs.index
                    idx = idx.append(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].obs.index
                    )
                    j = j + 1
                else:
                    idx_tmp = sc.pp.subsample(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == expr_label)],
                        n_obs=adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].shape[0],
                        copy=True,
                    ).obs.index
                    idx_tmp = idx_tmp.append(
                        adata_[(adata_.obs["leiden"] == i) & (adata_.obs[pert_key] == ref_label)].obs.index
                    )
                    idx = idx.append(idx_tmp)

        return idx

    def synergy(
        self,
        adata: AnnData,
        pert_key: str,
        base: str,
        A: str,
        B: str,
        AB: str,
        dim: int | None = 20,
        thres: float = 0.15,
        smoothness: float = 1e-4,
        preweight_label: str | None = None,
        **kwargs,
    ):
        """A wrapper for computing the synergy matrices.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            base: Control category from the `pert_key` column.
            A: the category for perturbation A.
            B: the category for perturbation B.
            AB: the category for the combinatorial perturbation A+B.
            dim: Use the first dim components in use_rep.
                If none, use a biwhitening procedure on the raw count matrix to derive a reasonable rank.
            thres: the threshold for the rank-dependence metric.
            smoothness: the coefficient determining the smooth level in entropic optimal transport problem.
            eps: Tolerate error of the optimal transport.
            preweight_label: the annotated label (e.g. cell type) that is used to assign weights for treated
                and control cells to balance across the label. Helps overcome the differential abundance issue.
            **kwargs: other parameters that can be passed to Cinemaot.causaleffect()

        Returns:
            Returns an AnnData object that contains the single-cell level synergy matrix de.X and the embedding.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dong_2023()
            >>> sc.pp.pca(adata)
            >>> model = pt.tl.Cinemaot()
            >>> combo = model.synergy(adata, pert_key='perturbation', base='No stimulation', A='IFNb', B='IFNg',
            >>>                   AB='IFNb+ IFNg', thres=0.5, smoothness=1e-5, eps=1e-3, solver='Sinkhorn')

        """
        adata1 = adata[adata.obs[pert_key].isin([base, A]), :].copy()
        adata2 = adata[adata.obs[pert_key].isin([B, AB]), :].copy()
        adata_link = adata[adata.obs[pert_key].isin([base, B]), :].copy()
        de1 = self.causaleffect(
            adata1,
            pert_key=pert_key,
            control=A,
            return_matching=True,
            dim=dim,
            thres=thres,
            smoothness=smoothness,
            preweight_label=preweight_label,
            **kwargs,
        )
        ot1 = de1.obsm["ot"]  # noqa: F841
        de2 = self.causaleffect(
            adata2,
            pert_key=pert_key,
            control=AB,
            return_matching=True,
            dim=dim,
            thres=thres,
            smoothness=smoothness,
            preweight_label=preweight_label,
            **kwargs,
        )
        ot2 = de2.obsm["ot"]  # noqa: F841
        de0 = self.causaleffect(
            adata_link,
            pert_key=pert_key,
            control=B,
            return_matching=True,
            dim=dim,
            thres=thres,
            smoothness=smoothness,
            preweight_label=preweight_label,
            **kwargs,
        )
        ot0 = de0.obsm["ot"]
        syn = sc.AnnData(
            np.array(-((ot0 / np.sum(ot0, axis=1)[:, None]) @ de2.X - de1.X)), obs=de1.obs.copy(), var=de1.var.copy()
        )
        syn.obsm["X_embedding"] = (ot0 / np.sum(ot0, axis=1)[:, None]) @ de2.obsm["X_embedding"] - de1.obsm[
            "X_embedding"
        ]
        return syn

    def attribution_scatter(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        cf_rep: str = "cf",
        use_raw: bool = False,
    ):
        """A simple function for computing confounder-specific genes.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            cf_rep: the place to put the confounder embedding in the original adata.obsm.
            use_raw: If true, use adata.raw.X to aggregate the pseudobulk profiles. Otherwise use adata.X.

        Returns:
            Returns the confounder effect (c_effect) and the residual effect (s_effect).

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> model = pt.tl.Cinemaot()
            >>> c_effect, s_effect = model.attribution_scatter(adata, pert_key="perturbation", control="No stimulation")
        """
        cf = adata.obsm[cf_rep]
        if use_raw:
            if issparse(adata.X):
                Y0 = adata.raw.X.toarray()[adata.obs[pert_key] == control, :]
                Y1 = adata.raw.X.toarray()[adata.obs[pert_key] != control, :]
            else:
                Y0 = adata.raw.X[adata.obs[pert_key] == control, :]
                Y1 = adata.raw.X[adata.obs[pert_key] != control, :]
        elif issparse(adata.X):
            Y0 = adata.X.toarray()[adata.obs[pert_key] == control, :]
            Y1 = adata.X.toarray()[adata.obs[pert_key] != control, :]
        else:
            Y0 = adata.X[adata.obs[pert_key] == control, :]
            Y1 = adata.X[adata.obs[pert_key] != control, :]
        X0 = cf[adata.obs[pert_key] == control, :]
        X1 = cf[adata.obs[pert_key] != control, :]
        ols0 = LinearRegression()
        ols0.fit(X0, Y0)
        ols1 = LinearRegression()
        ols1.fit(X1, Y1)
        c0 = ols0.predict(X0) - np.mean(ols0.predict(X0), axis=0)
        c1 = ols1.predict(X1) - np.mean(ols1.predict(X1), axis=0)
        e0 = Y0 - ols0.predict(X0)
        e1 = Y1 - ols1.predict(X1)
        c_effect = (np.linalg.norm(c1, axis=0) + 1e-6) / (np.linalg.norm(c0, axis=0) + 1e-6)
        s_effect = (np.linalg.norm(e1, axis=0) + 1e-6) / (np.linalg.norm(e0, axis=0) + 1e-6)
        return c_effect, s_effect

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_vis_matching(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        de: AnnData,
        pert_key: str,
        control: str,
        de_label: str,
        source_label: str,
        *,
        matching_rep: str = "ot",
        resolution: float = 0.5,
        normalize: str = "col",
        title: str = "CINEMA-OT matching matrix",
        min_val: float = 0.01,
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Visualize the CINEMA-OT matching matrix.

        Args:
            adata: the original anndata after running cinemaot.causaleffect or cinemaot.causaleffect_weighted.
            de: The anndata output from Cinemaot.causaleffect() or Cinemaot.causaleffect_weighted().
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            de_label: the label for differential response. If none, use leiden cluster labels at resolution 1.0.
            source_label: the confounder / cell type label.
            matching_rep: the place that stores the matching matrix. default de.obsm['ot'].
            resolution: Leiden resolution.
            normalize: normalize the coarse-grained matching matrix by row / column.
            title: the title for the figure.
            min_val: The min value to truncate the matching matrix.
            ax: Matplotlib axes object.
            {common_plot_args}
            **kwargs: Other parameters to input for seaborn.heatmap.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.cinemaot_example()
            >>> cot = pt.tl.Cinemaot()
            >>> de = cot.causaleffect(
            >>>         adata, pert_key="perturbation", control="No stimulation", return_matching=True,
            >>>         thres=0.5, smoothness=1e-5, eps=1e-3, solver="Sinkhorn", preweight_label="cell_type0528")
            >>> cot.plot_vis_matching(
            >>>         adata, de, pert_key="perturbation",control="No stimulation", de_label=None, source_label="cell_type0528")
        """
        adata_ = adata[adata.obs[pert_key] == control]

        df = pd.DataFrame(de.obsm[matching_rep])
        if de_label is None:
            de_label = "leiden"
            sc.pp.neighbors(de, use_rep="X_embedding")
            sc.tl.leiden(de, resolution=resolution)
        df["de_label"] = de.obs[de_label].astype(str).values
        df["de_label"] = "Response " + df["de_label"]
        df = df.groupby("de_label").sum().T
        df["source_label"] = adata_.obs[source_label].astype(str).values
        df = df.groupby("source_label").sum()

        df = df / df.sum(axis=0) if normalize == "col" else (df.T / df.sum(axis=1)).T
        df = df.clip(lower=min_val) - min_val
        df = df / df.sum(axis=0) if normalize == "col" else (df.T / df.sum(axis=1)).T

        g = heatmap(df, annot=True, ax=ax, **kwargs)
        plt.title(title)

        if return_fig:
            return g
        plt.show()
        return None


class Xi:
    """A fast implementation of cross-rank dependence metric used in CINEMA-OT."""

    def __init__(self, x, y, random_state: int | None = 0):
        self.x = x
        self.y = y
        self.random_state = random_state

    @property
    def sample_size(self):
        return len(self.x)

    @property
    def x_ordered_rank(self):
        # PI is the rank vector for x, with ties broken at random
        len_x = len(self.x)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len_x)
        x_shuffled = self.x[perm]

        ranks = np.empty(len_x, dtype=int)
        ranks[perm[np.argsort(x_shuffled, stable=True)]] = np.arange(1, len_x + 1)

        return ranks

    @property
    def y_rank_max(self):
        # f[i] is number of j s.t. y[j] <= y[i], divided by n.
        return ss.rankdata(self.y, method="max") / self.sample_size

    @property
    def g(self):
        # g[i] is number of j s.t. y[j] >= y[i], divided by n.
        return ss.rankdata(-self.y, method="max") / self.sample_size

    @property
    def x_ordered(self):
        # order of the x's, ties broken at random.
        return np.argsort(self.x_ordered_rank)

    @property
    def x_rank_max_ordered(self):
        return self.y_rank_max[self.x_ordered]

    @property
    def mean_absolute(self):
        x1 = self.x_rank_max_ordered[0 : (self.sample_size - 1)]
        x2 = self.x_rank_max_ordered[1 : self.sample_size]

        return np.mean(np.abs(x1 - x2)) * (self.sample_size - 1) / (2 * self.sample_size)

    @property
    def inverse_g_mean(self):
        gvalue = self.g
        return np.mean(gvalue * (1 - gvalue))

    @property
    def correlation(self):
        """Xi correlation."""
        return 1 - self.mean_absolute / self.inverse_g_mean

    @classmethod
    def xi(cls, x, y):
        return cls(x, y)

    def pval_asymptotic(self, ties: bool = False):
        """Returns p values of the correlation.

        Args:
            ties: boolean
                If ties is true, the algorithm assumes that the data has ties
                and employs the more elaborated theory for calculating
                the P-value. Otherwise, it uses the simpler theory. There is
                no harm in setting tiles True, even if there are no ties.

        Returns:
            p value
        """
        # If there are no ties, return xi and theoretical P-value:
        if ties:
            return 1 - ss.norm.cdf(np.sqrt(self.sample_size) * self.correlation / np.sqrt(2 / 5))

        # If there are ties, and the theoretical method is to be used for calculation P-values:
        # The following steps calculate the theoretical variance in the presence of ties:
        sorted_ordered_x_rank = sorted(self.x_rank_max_ordered)

        ind = [i + 1 for i in range(self.sample_size)]
        ind2 = [2 * self.sample_size - 2 * ind[i - 1] + 1 for i in ind]

        a = np.mean([i * j * j for i, j in zip(ind2, sorted_ordered_x_rank, strict=False)]) / self.sample_size

        c = np.mean([i * j for i, j in zip(ind2, sorted_ordered_x_rank, strict=False)]) / self.sample_size

        cq = np.cumsum(sorted_ordered_x_rank)

        m = [
            (i + (self.sample_size - j) * k) / self.sample_size
            for i, j, k in zip(cq, ind, sorted_ordered_x_rank, strict=False)
        ]

        b = np.mean([np.square(i) for i in m])
        v = (a - 2 * b + np.square(c)) / np.square(self.inverse_g_mean)

        return 1 - ss.norm.cdf(np.sqrt(self.sample_size) * self.correlation / np.sqrt(v))


class SinkhornKnopp:
    """An simple implementation of Sinkhorn iteration used in the biwhitening approach."""

    def __init__(self, max_iter: float = 1000, setr: int = 0, setc: float = 0, epsilon: float = 1e-3):
        if max_iter < 0:
            raise ValueError(f"max_iter is {max_iter} but must be greater than 0.")
        self._max_iter = int(max_iter)

        if not epsilon > 0 and epsilon < 1:
            raise ValueError(f"epsilon is {epsilon} but must be between 0 and 1 exclusive.")
        self._epsilon = epsilon
        self._setr = setr
        self._setc = setc
        self._stopping_condition: str | None = None
        self._iterations = 0
        self._D1 = np.ones(1)
        self._D2 = np.ones(1)

    def fit(self, P: ArrayLike):
        """Fit the diagonal matrices in Sinkhorn Knopp's algorithm.

        Args:
            P: 2d array-like
               Must be a square non-negative 2d array-like object, that
               is convertible to a numpy array. The matrix must not be
               equal to 0 and it must have total support for the algorithm to converge.

        Returns:
            A double stochastic matrix.
        """
        P = np.asarray(P)
        assert np.all(P >= 0)
        assert P.ndim == 2

        N = P.shape[0]
        rsum = P.shape[1] if np.sum(abs(self._setr)) == 0 else self._setr
        csum = P.shape[0] if np.sum(abs(self._setc)) == 0 else self._setc
        max_threshr = rsum + self._epsilon
        min_threshr = rsum - self._epsilon
        max_threshc = csum + self._epsilon
        min_threshc = csum - self._epsilon
        # Initialize r and c, the diagonals of D1 and D2
        # and warn if the matrix does not have support.
        r = np.ones((N, 1))
        pdotr = P.T.dot(r)

        c = 1 / pdotr
        pdotc = P.dot(c)

        r = 1 / pdotc
        del pdotr, pdotc

        P_eps = np.copy(P)
        while (
            np.any(np.sum(P_eps, axis=1) < min_threshr)
            or np.any(np.sum(P_eps, axis=1) > max_threshr)
            or np.any(np.sum(P_eps, axis=0) < min_threshc)
            or np.any(np.sum(P_eps, axis=0) > max_threshc)
        ):
            c = csum / P.T.dot(r)
            r = rsum / P.dot(c)

            self._D1 = np.diag(np.squeeze(r))
            self._D2 = np.diag(np.squeeze(c))

            P_eps = np.diag(self._D1)[:, None] * P * np.diag(self._D2)[None, :]

            self._iterations += 1

            if self._iterations >= self._max_iter:
                self._stopping_condition = "max_iter"
                break

        if not self._stopping_condition:
            self._stopping_condition = "epsilon"

        self._D1 = np.diag(np.squeeze(r))
        self._D2 = np.diag(np.squeeze(c))
        P_eps = np.diag(self._D1)[:, None] * P * np.diag(self._D2)[None, :]

        return P_eps
