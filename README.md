[![Build](https://github.com/scverse/pertpy/actions/workflows/build.yml/badge.svg)](https://github.com/scverse/pertpy/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/scverse/pertpy/graph/badge.svg?token=1dTpIPBShv)](https://codecov.io/gh/scverse/pertpy)
[![License](https://img.shields.io/github/license/scverse/pertpy)](https://opensource.org/licenses/Apache2.0)
[![PyPI](https://img.shields.io/pypi/v/pertpy.svg)](https://pypi.org/project/pertpy/)
[![Python Version](https://img.shields.io/pypi/pyversions/pertpy)](https://pypi.org/project/pertpy)
[![Read the Docs](https://img.shields.io/readthedocs/pertpy/latest.svg?label=Read%20the%20Docs)](https://pertpy.readthedocs.io/)
[![Test](https://github.com/scverse/pertpy/actions/workflows/test.yml/badge.svg)](https://github.com/scverse/pertpy/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# pertpy - Perturbation Analysis in Python

Pertpy is a scverse ecosystem framework for analyzing large-scale single-cell perturbation experiments.
It provides tools for harmonizing perturbation datasets, automating metadata annotation, calculating perturbation distances, and efficiently analyzing how cells respond to various stimuli like genetic modifications, drug treatments, and environmental changes.

![fig1](https://github.com/user-attachments/assets/d2e32d69-b767-4be3-a938-77a9dce45d3f)

## Documentation

Please read the [documentation](https://pertpy.readthedocs.io/en/latest) for installation, tutorials, use cases, and more.

## Installation

We recommend installing and running pertpy on a recent version of Linux (e.g. Ubuntu 24.04 LTS).
No particular hardware beyond a standard laptop is required.

You can install _pertpy_ in less than a minute via [pip] from [PyPI]:

```console
pip install pertpy
```

or [conda-forge]:

```console
conda install -c conda-forge pertpy
```

### Differential gene expression

If you want to use the differential gene expression interface, please install pertpy by running:

```console
pip install 'pertpy[de]'
```

### tascCODA

if you want to use tascCODA, please install pertpy as follows:

```console
pip install 'pertpy[tcoda]'
```

### milo

milo requires either the "de" extra for the "pydeseq2" solver:

```console
pip install 'pertpy[de]'
```

or, edger, statmod, and rpy2 for the "edger" solver:

```R
BiocManager::install("edgeR")
BiocManager::install("statmod")
```

```console
pip install rpy2
```

## Citation

```bibtex
@article{Heumos2025,
  author    = {Heumos, Lukas and Ji, Yuge and May, Lilly and Green, Tessa D. and Peidli, Stefan and Zhang, Xinyue and Wu, Xichen and Ostner, Johannes and Schumacher, Antonia and Hrovatin, Karin and Müller, Michaela and Chong, Faye and Sturm, Gregor and Tejada, Alejandro and Dann, Emma and Dong, Mingze and Pinto, Gonçalo and Bahrami, Mojtaba and Gold, Ilan and Rybakov, Sergei and Namsaraeva, Altana and Moinfar, Amir Ali and Zheng, Zihe and Roellin, Eljas and Mekki, Isra and Sander, Chris and Lotfollahi, Mohammad and Schiller, Herbert B. and Theis, Fabian J.},
  title     = {Pertpy: an end-to-end framework for perturbation analysis},
  journal   = {Nature Methods},
  year      = {2025},
  publisher = {Springer Nature},
  doi       = {10.1038/s41592-025-02909-7},
  url       = {https://doi.org/10.1038/s41592-025-02909-7},
  issn      = {1548-7105}
}
```

[pip]: https://pip.pypa.io/
[pypi]: https://pypi.org/
[api]: https://pertpy.readthedocs.io/en/latest/api.html
[conda-forge]: https://anaconda.org/conda-forge/pertpy
[//]: # "numfocus-fiscal-sponsor-attribution"

pertpy is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>
