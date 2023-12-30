# # turn the list of gene IDs to a boolean mask of var_names
# if layer is not None:
#     mtx = adata.layers[layer]
# else:
#     mtx = adata.X
# for drug in targets:
#     targets[drug] = np.isin(var_names, targets[drug])
# # perform scoring
# # the scoring shall be done via matrix multiplication
# # of the original cell by gene matrix, by a new gene by drug matrix
# # with the entries in the new matrix being the weights of each gene for that drug
# # the first part, the mean across targets, is constant; prepare weights for that
# weights = pd.DataFrame(targets, index=var_names)
# # kick out drugs with no targets
# weights = weights.loc[:, weights.sum() > 0]
# # scale to 1 sum for each column, weights for mean acquired. get mean
# weights = weights / weights.sum()
# if issparse(X):
#     scores = X.dot(weights)
# else:
#     scores = np.dot(X, weights)
# # we now have the final form of the scores
# # create a little helper adata thingy based on them
# # store existing .obsm in there for ease of plotting stuff
# adata.uns['drug2cell'] = anndata.AnnData(scores, obs=adata.obs)
# adata.uns['drug2cell'].var_names = weights.columns
# adata.uns['drug2cell'].obsm = adata.obsm
# # store gene group membership, going back to targets for it
# for drug in weights.columns:
#     # mask the var_names with the membership, and join into a single delimited string
#     adata.uns['drug2cell'].var.loc[drug, 'genes'] = sep.join(var_names[targets[drug]])
#     # pull out the old full membership dict and store that too
#     adata.uns['drug2cell'].var.loc[drug, 'all_genes'] = sep.join(full_targets[drug])
