import scanpy as sc
import numpy as np
from pytest import fixture
from pandas import DataFrame

import pertpy as pt

class TestPermutationTest:
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
    
    def test_Etest(self, adata):
        etest = pt.tl.PermutationTest('edistance', n_perms=1000, 
                                      obsm_key='X_pca', alpha=0.05, 
                                      correction='holm-sidak')
        tab = etest(adata, groupby='perturbation', contrast='control')
        # Well-defined output
        assert tab.shape[1] == 5
        assert type(tab) == DataFrame
        # p-values are in [0,1]
        assert tab['pvalue'].min() >= 0
        assert tab['pvalue'].max() <= 1
        assert tab['pvalue_adj'].min() >= 0
        assert tab['pvalue_adj'].max() <= 1
    
    def test_pb_test(self, adata):
        pb_test = pt.tl.PermutationTest('pseudobulk', n_perms=1000, 
                                      obsm_key='X_pca', alpha=0.05, 
                                      correction='holm-sidak')
        tab = pb_test(adata, groupby='perturbation', contrast='control')
        # Well-defined output
        assert tab.shape[1] == 5
        assert type(tab) == DataFrame
        # p-values are in [0,1]
        assert tab['pvalue'].min() >= 0
        assert tab['pvalue'].max() <= 1
        assert tab['pvalue_adj'].min() >= 0
        assert tab['pvalue_adj'].max() <= 1
