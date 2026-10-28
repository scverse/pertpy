# Quickstart

Every pertpy tool follows the same pattern: load an {class}`~anndata.AnnData` object, create the tool, and call its methods on the data.
The example below loads a small Perturb-seq dataset, computes the E-distance between all perturbations and tests which perturbations differ from the control cells.

```python
import pertpy as pt

adata = pt.dt.distance_example()

distance = pt.tl.Distance("edistance", obsm_key="X_pca")
pairwise = distance.pairwise(adata, groupby="perturbation")

test = pt.tl.DistanceTest("edistance", n_perms=100, obsm_key="X_pca")
results = test(adata, groupby="perturbation", contrast="control")
```

Datasets live in `pt.dt`, preprocessing in `pt.pp`, tools in `pt.tl` and metadata annotation in `pt.md`.
Some tools need optional dependencies, which {doc}`installation` lists.

## Find the right tool

| Question                                                   | Tools                                                                                                                                         | Tutorial                                                                              |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Which guide RNAs did each cell receive?                    | {class}`~pertpy.preprocessing.GuideAssignment`                                                                                                | {doc}`tutorials/notebooks/guide_rna_assignment`                                       |
| Which cells were actually perturbed?                       | {class}`~pertpy.tools.Mixscape`, {class}`~pertpy.tools.Mixscale`                                                                              | {doc}`tutorials/notebooks/perturbation_efficacy`                                      |
| Which genes change between conditions?                     | {class}`~pertpy.tools.PyDESeq2`, {class}`~pertpy.tools.EdgeR`, {class}`~pertpy.tools.Statsmodels`, {class}`~pertpy.tools.WilcoxonTest`        | {doc}`tutorials/notebooks/differential_gene_expression`                               |
| Did the cell type composition change?                      | {class}`~pertpy.tools.Sccoda`, {class}`~pertpy.tools.Tasccoda`                                                                                | {doc}`tutorials/notebooks/sccoda`, {doc}`tutorials/notebooks/tasccoda`                |
| Did cell states change in abundance, without cell types?   | {class}`~pertpy.tools.Milo`                                                                                                                   | {doc}`tutorials/notebooks/milo`                                                       |
| Which cell types respond most strongly?                    | {class}`~pertpy.tools.Augur`                                                                                                                  | {doc}`tutorials/notebooks/augur`                                                      |
| Which programs are coordinated across cell types?          | {class}`~pertpy.tools.Dialogue`                                                                                                               | {doc}`tutorials/notebooks/dialogue`                                                   |
| Which pathways or drug targets are enriched?               | {class}`~pertpy.tools.Enrichment`                                                                                                             | {doc}`tutorials/notebooks/enrichment`                                                 |
| How different are two perturbations, and is it significant? | {class}`~pertpy.tools.Distance`, {class}`~pertpy.tools.DistanceTest`                                                                         | {doc}`tutorials/notebooks/distances`, {doc}`tutorials/notebooks/distance_tests`       |
| How do perturbations compare as a whole?                   | {class}`~pertpy.tools.PseudobulkSpace`, {class}`~pertpy.tools.CentroidSpace`                                                                  | {doc}`tutorials/notebooks/perturbation_space`                                         |
| What is the effect after removing confounding variation?   | {class}`~pertpy.tools.Cinemaot`                                                                                                               | {doc}`tutorials/notebooks/cinemaot`                                                   |
| How would an unseen cell type respond?                     | {class}`~pertpy.tools.Scgen`                                                                                                                  | {doc}`tutorials/notebooks/scgen_perturbation_prediction`                              |
| What are the cell lines, compounds and drugs in my data?   | {class}`~pertpy.metadata.CellLine`, {class}`~pertpy.metadata.Compound`, {class}`~pertpy.metadata.Drug`, {class}`~pertpy.metadata.Moa`          | {doc}`tutorials/notebooks/metadata_annotation`                                        |

The {doc}`usecases` combine several of these tools on published perturbation studies.
