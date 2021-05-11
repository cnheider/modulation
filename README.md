
<!--![header](.github/images/header.png)-->

<p align="center">
  <img src=".github/images/header.png" alt='header' />
</p>

<h1 align="center">Audition</h1>

<!--# Audition-->

This repository will host implementation audio modality algorithms applied to the [Neodroid](https://github.com/sintefneodroid/) platform.

---

_[Neodroid](https://github.com/sintefneodroid) is developed with support from Research Council of Norway Grant #262900. ([https://www.forskningsradet.no/prosjektbanken/#/project/NFR/262900](https://www.forskningsradet.no/prosjektbanken/#/project/NFR/262900))_

---


[![PyPI Status](https://badge.fury.io/py/repo.svg)](https://badge.fury.io/py/repo)
[![Python Versions](https://img.shields.io/pypi/pyversions/repo.svg)](https://pypi.org/project/repo/)
[![PyPI Status](https://pepy.tech/badge/repo)](https://pepy.tech/project/repo)
[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fauthor%2Frepo%2Fbadge%3Fref%3Dmaster&style=flat)](https://actions-badge.atrox.dev/author/repo/goto?ref=master)
[![Documentation Status](https://readthedocs.org/projects/repo/badge/?version=latest)](https://lhotse.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/author/repo/branch/master/graph/badge.svg)](https://codecov.io/gh/author/repo)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/author/repo/blob/master/notebooks/repo-introduction.ipynb)

| [![Build Status](https://travis-ci.org/sintefneodroid/agent.svg?branch=master)](https://travis-ci.org/sintefneodroid/agent)  | [![Coverage Status](https://coveralls.io/repos/github/sintefneodroid/agent/badge.svg?branch=master)](https://coveralls.io/github/sintefneodroid/agent?branch=master)  | [![GitHub Issues](https://img.shields.io/github/issues/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/issues)  |  [![GitHub Forks](https://img.shields.io/github/forks/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/network) | [![GitHub Stars](https://img.shields.io/github/stars/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/stargazers) |[![GitHub License](https://img.shields.io/github/license/sintefneodroid/agent.svg?style=flat)](https://github.com/sintefneodroid/agent/blob/master/LICENSE.md) | [![CodeScene System Mastery](https://codescene.io/projects/12883/status-badges/system-mastery)](https://codescene.io/projects/12883) | [![CodeScene Code Health](https://codescene.io/projects/12883/status-badges/code-health)](https://codescene.io/projects/12883) |
|---|---|---|---|---|---|---|---|

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
    - [Segmentation](#segmentation)
- [Contributing](#contributing)
- [Other Components](#other-components-of-the-neodroid-platform)

# Algorithms

- [SAGAN](vision/.py) - Generative model
- [CVAE](vision/.py) - Generative model
- [BVAE](vision/.py) - Generative model
- [Vanilla-CNN](vision/.py) - Classification
- [Vanilla-MLP](vision/.py) - Classification
- [YOLO](vision/.py) - Segmentation

# Requirements

- pytorch
- tqdm
- Pillow
- numpy
- matplotlib
- torchvision
- torch
- Neodroid
- pynput

To install these use the command:

````bash
pip3 install -r requirements.txt
````

# Usage

Export python path to the repo root so we can use the utilities module

````bash
export PYTHONPATH=/path-to-repo/
````

For training a agent use:

````bash
python3 procedures/train_agent.py
````

For testing a trained agent use:

````bash
python3 procedures/test_agent.py
````

# Results

## Segmentation

[Code](samples/regression/segmentation/run.py)

### Screenshots

![Segmentation](.github/images/results/ori_mask_seg_recon.png)

# Contributing

See guidelines for contributing [here](CONTRIBUTING.md).

# Citation

For citation you may use the following bibtex entry:

````
@misc{neodroid-vision,
  author = {Heider, Christian},
  title = {Neodroid Vision},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aivclab/vision}},
}
````

# Authors

* **Christian Heider Nielsen** - [cnheider](https://github.com/cnheider)

Here other [contributors](https://github.com/aivclab/vision/contributors) to this project are listed.
