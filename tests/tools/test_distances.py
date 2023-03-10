import scanpy as sc
import numpy as np
from pytest import fixture
from pandas import DataFrame

import pertpy as pt

distances = ['edistance', 'pseudobulk', 'mean_pairwise']

class TestDistances:
    @fixture
    def adata(self):
        adata = pt.dt.dixit_2016_scperturb()
        obs_key = 'perturbation'
        
        adata.layers['counts'] = adata.X.copy()

        # basic qc and pp
        sc.pp.filter_cells(adata, min_counts=1000)
        sc.pp.normalize_per_cell(adata)
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.log1p(adata)

        # subsample against high class imbalance
        N_min = 100
        counts = adata.obs[obs_key].value_counts()
        groups = counts.index[counts>=N_min]
        indices = [np.random.choice(adata.obs_names[adata.obs[obs_key]==group], size=N_min, replace=False) for group in groups]
        selection = np.hstack(np.array(indices))
        adata = adata[selection].copy()
        sc.pp.filter_genes(adata, min_cells=3)  # sanity cleaning

        # select HVGs
        n_var_max = 2000  # max total features to select
        sc.pp.highly_variable_genes(adata, n_top_genes=n_var_max, subset=False, flavor='seurat_v3', layer='counts')
        sc.pp.pca(adata, use_highly_variable=True)
        return adata
    
    def test_distance_axioms(self, adata):
        for distance_ in distances:
            # Test if distances are well-defined in accordance with metric axioms
            distance = pt.tl.Distance(distance_, 'X_pca')
            df = distance.pairwise(adata, groupby='perturbation', verbose=True)
            # (M1) Positiv definiteness
            assert all(np.diag(df.values) == 0)  # distance to self is 0
            assert len(df) == np.sum(df.values == 0)  # distance to other is not 0
            # (M2) Symmetry
            assert np.sum(df.values - df.values.T) == 0
            assert df.columns.equals(df.index)
            # (M3) Triangle inequality (we just probe this for a few random triplets)
            for i in range(100):
                triplet = np.random.choice(df.index, size=3, replace=False)
                assert df.loc[triplet[0], triplet[1]] + df.loc[triplet[1], triplet[2]] >= df.loc[triplet[0], triplet[2]]
        
        