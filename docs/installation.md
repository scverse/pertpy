```{highlight} shell

```

# Installation

## Stable release

To install pertpy, run this command in your terminal:

```console
pip install pertpy
```

This is the preferred method to install pertpy, as it will always install the most recent stable release.

If you don't have [pip] installed, this [Python installation guide] can guide you through the process.

## Additional dependency groups

### scCODA and tascCODA

TascCODA requires an additional set of dependencies (ete4, pyqt6, and toytree) that can be installed by running:

```console
pip install pertpy[tcoda]
```

## Differential gene expression interface

The DGE interface of pertpy requires additional dependencies that can be installed by running:

```console
pip install pertpy[de]
```

Note that edger in pertpy requires edger and rpy2 to be installed:

```R
BiocManager::install("edgeR")
```

```console
pip install rpy2
```

## From sources

The sources for pertpy can be downloaded from the [Github repo].
Please note that you require [poetry] to be installed.

You can either clone the public repository:

```console
$ git clone git://github.com/scverse/pertpy
```

Or download the [tarball]:

```console
$ curl -OJL https://github.com/scverse/pertpy/tarball/master
```

## Apple Silicon

If you want to install and use pertpy on a machine with macOS and M-Chip, the installation is slightly more complex.
This is because pertpy depends on [scvi-tools], which can currently only run on Apple Silicon machines when installed
using a native python version (due to a dependency on jax, which cannot be run via Rosetta).

Follow these steps to install pertpy on an Apple Silicon machine (tested on a MacBook Pro with M1 chip and macOS 14.0):

1. Install [Homebrew]

2. Install Apple Silicon version of Mambaforge (If you already have Anaconda/Miniconda installed, make sure
   having both mamba and conda won't cause conflicts)

    ```console
    brew install --cask mambaforge
    ```

3. Create a new environment using mamba and activate it

    ```console
    mamba create -n pertpy-env
    mamba activate pertpy-env
    ```

4. Clone the GitHub Repository

    ```console
    git clone https://github.com/scverse/pertpy.git
    ```

5. Go inside the pertpy folder and install pertpy

    ```console
    cd pertpy
    pip install .
    ```

Now you're ready to use pertpy as usual within the environment (`import pertpy`).

[github repo]: https://github.com/scverse/pertpy
[pip]: https://pip.pypa.io
[poetry]: https://python-poetry.org/
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
[tarball]: https://github.com/scverse/pertpy/tarball/master
[scvi-tools]: https://docs.scvi-tools.org/en/latest/installation.html
[Homebrew]: https://brew.sh/
