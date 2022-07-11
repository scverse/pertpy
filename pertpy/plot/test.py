'''
import muon as mu
import scanpy as sc



mdata = mu.read('/Users/xinyuezhang/pertpy/mdata_processed.h5mu')


adata_adt = mdata['adt'].copy()
adata_adt.obs['mixscape_class_global'] = mdata['rna'].obs['mixscape_class_global']


gene_target_mask = (mdata['adt'].obs['gene_target'] == "NT") | (mdata['adt'].obs['gene_target'] == "JAK2")| (mdata['adt'].obs['gene_target'] == "STAT1")| (mdata['adt'].obs['gene_target'] == "IFNGR1")| (mdata['adt'].obs['gene_target'] == "IFNGR2")| (mdata['adt'].obs['gene_target'] == "IRF1")
adata_adt_subset = adata_adt[gene_target_mask].copy()


sc.pl.violin(adata_adt_subset, keys=['PDL1'], groupby='gene_target', hue='mixscape_class_global')
'''

import pertpy as pt
import muon as mu
import pandas as pd
import scanpy as sc
import warnings
mdata = mu.read('/Users/xinyuezhang/pertpy/mdata_processed.h5mu')
mdata['adt'].obs['mixscape_class_global'] = mdata['rna'].obs['mixscape_class_global']
#pt.pl.mixscape_violin(adata = mdata['adt'], target_gene_idents=["NT","JAK2","STAT1","IFNGR1","IFNGR2", "IRF1"], keys='PDL1', groupby='gene_target', hue='mixscape_class_global')
pt.pl.plotperturbscore(adata = mdata['rna'], labels='gene_target', target_gene='IFNGR2', color = 'orange')
