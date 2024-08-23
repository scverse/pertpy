[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build](https://github.com/scverse/pertpy/actions/workflows/build.yml/badge.svg)](https://github.com/scverse/pertpy/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/scverse/pertpy/graph/badge.svg?token=1dTpIPBShv)](https://codecov.io/gh/scverse/pertpy)
[![License](https://img.shields.io/github/license/scverse/pertpy)](https://opensource.org/licenses/Apache2.0)
[![PyPI](https://img.shields.io/pypi/v/pertpy.svg)](https://pypi.org/project/pertpy/)
[![Python Version](https://img.shields.io/pypi/pyversions/pertpy)](https://pypi.org/project/pertpy)
[![Read the Docs](https://img.shields.io/readthedocs/pertpy/latest.svg?label=Read%20the%20Docs)](https://pertpy.readthedocs.io/)
[![Test](https://github.com/scverse/pertpy/actions/workflows/test.yml/badge.svg)](https://github.com/scverse/pertpy/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

# pertpy

![fig1](https://github.com/user-attachments/assets/d2e32d69-b767-4be3-a938-77a9dce45d3f)

## Documentation

Please read the [documentation](https://pertpy.readthedocs.io/en/latest).

## Installation

We recommend installing and running pertpy on a recent version of Linux (e.g. Ubuntu 24.04 LTS).
No particular hardware beyond a standard laptop is required.

You can install _pertpy_ in less than a minute via [pip] from [PyPI]:

```console
pip install pertpy
```

if you want to use scCODA or tascCODA, please install pertpy as follows:

```console
pip install 'pertpy[coda]'
```

If you want to use the differential gene expression interface, please install pertpy by running:

```console
pip install 'pertpy[de]'
```

## Citation

[Lukas Heumos, Yuge Ji, Lilly May, Tessa Green, Xinyue Zhang, Xichen Wu, Johannes Ostner, Stefan Peidli, Antonia Schumacher, Karin Hrovatin, Michaela Mueller, Faye Chong, Gregor Sturm, Alejandro Tejada, Emma Dann, Mingze Dong, Mojtaba Bahrami, Ilan Gold, Sergei Rybakov, Altana Namsaraeva, Amir Ali Moinfar, Zihe Zheng, Eljas Roellin, Isra Mekki, Chris Sander, Mohammad Lotfollahi, Herbert B. Schiller, Fabian J. Theis
bioRxiv 2024.08.04.606516; doi: https://doi.org/10.1101/2024.08.04.606516](https://www.biorxiv.org/content/10.1101/2024.08.04.606516v1)

[pip]: https://pip.pypa.io/
[pypi]: https://pypi.org/
[usage]: https://pertpy.readthedocs.io/en/latest/usage/usage.html
