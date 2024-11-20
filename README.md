# scmallet

Python wrapper of MALLET for LDA analysis on single-cell data.

MALLET is the LDA backend chosen by pycistopic. The implementation in this package has several difference than pycistopic:

1. Improved paralization.
2. Allow train LDA model with cell subset and then inference ramaining cells. This is nicely supported by MALLET itself.
3. Allow the training process to be resumable.
4. Work with anndata.

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install scmallet:

1. Install the latest release of `scmallet` from [PyPI](https://pypi.org/project/scmallet/):

```bash
pip install scmallet
```

2. Install the latest development version:

```bash
pip install git+https://github.com/lhqing/scmallet.git@main
```

## Usage

See example usage here: https://github.com/lhqing/scmallet/blob/main/tests/example.ipynb

## Citation

-   Mallet Package: https://mimno.github.io/Mallet/

    > McCallum, Andrew Kachites. "MALLET: A Machine Learning for Language Toolkit." http://mallet.cs.umass.edu. 2002.

-   PyCistopic: https://github.com/aertslab/pycisTopic
    > Bravo Gonzalez-Blas, C. & De Winter, S. et al. (2022). SCENIC+: single-cell multiomic inference of enhancers and gene regulatory networks
