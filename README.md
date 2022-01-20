
<!--![header](.github/images/header.png)-->

<h1 align="center">Modulation</h1>

<!--# Modulation-->

This repository will host implementation time series signals modality algorithms.

[![PyPI Status](https://badge.fury.io/py/repo.svg)](https://badge.fury.io/py/repo)
[![Python Versions](https://img.shields.io/pypi/pyversions/repo.svg)](https://pypi.org/project/repo/)
[![PyPI Status](https://pepy.tech/badge/repo)](https://pepy.tech/project/repo)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fauthor%2Frepo%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/author/repo/goto?ref=master)
[![Documentation Status](https://readthedocs.org/projects/repo/badge/?version=latest)](https://lhotse.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/author/repo/branch/master/graph/badge.svg)](https://codecov.io/gh/author/repo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/author/repo/blob/master/notebooks/repo-introduction.ipynb)

| [![Build Status](https://travis-ci.org/aivclab/modulation.svg?branch=master)](https://travis-ci.org/aivclab/modulation) | [![Coverage Status](https://coveralls.io/repos/github/aivclab/modulation/badge.svg?branch=master)](https://coveralls.io/github/aivclab/modulation?branch=master) | [![GitHub Issues](https://img.shields.io/github/issues/aivclab/modulation.svg?style=flat)](https://github.com/aivclab/modulation/issues)  |  [![GitHub Forks](https://img.shields.io/github/forks/aivclab/modulation.svg?style=flat)](https://github.com/aivclab/modulation/network) | [![GitHub Stars](https://img.shields.io/github/stars/aivclab/modulation.svg?style=flat)](https://github.com/aivclab/modulation/stargazers) |[![GitHub License](https://img.shields.io/github/license/aivclab/modulation.svg?style=flat)](https://github.com/aivclab/modulation/blob/master/LICENSE.md) | [![CodeScene System Mastery](https://codescene.io/projects/12883/status-badges/system-mastery)](https://codescene.io/projects/12883) | [![CodeScene Code Health](https://codescene.io/projects/12883/status-badges/code-health)](https://codescene.io/projects/12883) |
|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|---|---|---|---|---|

<p align="center" width="100%">
  <a href="https://www.python.org/">
    <img alt="python" src=".github/images/python.svg" height="40" align="left">
  </a>
  <a href="http://pytorch.org/"style="float: right;">
    <img alt="pytorch" src=".github/images/pytorch.svg" height="40" align="right" >
  </a>
</p>
<p align="center" width="100%">
  <a href="http://www.numpy.org/">
    <img alt="numpy" src=".github/images/numpy.svg" height="40" align="left">
  </a>
  <a href="https://github.com/tqdm/tqdm" style="float:center;">
    <img alt="tqdm" src=".github/images/tqdm.gif" height="40" align="center">
  </a>
</p>

# Contents Of This Readme

- [Algorithms](#algorithms)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
    - [Denoise](#denoise)
- [Contributing](#contributing)


# Algorithms

- [spectral_denoise](modulation/regression/spectral_denoise.py) - Spectral Denoise


# Requirements

- pytorch
- tqdm
- Pillow
- numpy
- matplotlib
- torchaudio
- torch
- pynput

To install these use the command:

````bash
pip3 install -r requirements.txt
````

# Usage

Export python path to the repo root, so we can use the utilities module

````bash
export PYTHONPATH=/path-to-repo/
````
# Results

## Denoise

[Code](samples/regression/denoise.py)

### Screenshots

![Denoise](.github/images/results/denoise.png)

# Contributing

See guidelines for contributing [here](CONTRIBUTING.md).

# Citation

For citation you may use the following bibtex entry:

````
@misc{modulation,
  author = {Heider, Christian},
  title = {Neodroid Vision},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aivclab/modulation}},
}
````

# Authors

* **Christian Heider Nielsen** - [cnheider](https://github.com/cnheider)

Here other [contributors](https://github.com/aivclab/modulation/contributors) to this project are listed.
