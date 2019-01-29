<p align="center">
  <img src="docs/img/saber_logo.png", style="height:150px">
</p>

<h1 align="center">
  Saber
</h1>

<p align="center">
  <a href="https://travis-ci.org/BaderLab/saber">
    <img src="https://travis-ci.org/BaderLab/saber.svg?branch=master"
         alt="Travis CI">
  </a>
  <a href="https://www.codacy.com/app/JohnGiorgi/Saber?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BaderLab/saber&amp;utm_campaign=Badge_Grade">
    <img src="https://api.codacy.com/project/badge/Grade/d122e87152d84f959ee6d97b71d616cb" alt='Codacy Status'/>
  </a>
  <a href='https://coveralls.io/github/BaderLab/saber?branch=master'>
    <img src='https://coveralls.io/repos/github/BaderLab/saber/badge.svg?branch=master' alt='Coverage Status'/>
  </a>
  <a href='https://spacy.io'>
    <img src='https://img.shields.io/badge/spaCy-v2-09a3d5.svg' alt='Spacy Version'/>
  </a>
  <a href='http://makeapullrequest.com'>
    <img src='https://img.shields.io/badge/PRs-welcome-blue.svg?style=shields' alt='PRs Welcome'/>
  </a>
  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/>
  </a>
    <a href='https://colab.research.google.com/drive/1WD7oruVuTo6p_908MQWXRBdLF3Vw2MPo'>
    <img src='https://img.shields.io/badge/launch-Google%20Colab-orange.svg' alt='Colab'/>
  </a>
</p>

<p align="center"><b>Saber</b> (<b>S</b>equence <b>A</b>nnotator for <b>B</b>iomedical <b>E</b>ntities and <b>R</b>elations) is a deep-learning based tool for <b>information extraction</b> in the biomedical domain.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#documentation">Documentation</a>
</p>

## Installation

To install Saber, you will need `python3.6`.

### Latest PyPI stable release

[![PyPI-Status](https://img.shields.io/pypi/v/saber.svg?colorB=blue)](https://pypi.org/project/saber/)
[![PyPI-Downloads](https://img.shields.io/pypi/dm/saber.svg?colorB=blue&logo=python&logoColor=white)](https://pypi.org/project/saber)
[![Libraries-Dependents](https://img.shields.io/librariesio/dependent-repos/pypi/saber.svg?colorB=blue&logo=koding&logoColor=white)](https://github.com/baderlab/saber/network/dependents)

```sh
(saber) $ pip install saber
```

> The install from PyPI is currently broken, please install using the instructions below.

### Latest development release on GitHub

[![GitHub-Status](https://img.shields.io/github/tag-date/baderlab/saber.svg?logo=github)](https://github.com/baderlab/saber/releases)
[![GitHub-Stars](https://img.shields.io/github/stars/baderlab/saber.svg?logo=github&label=stars)](https://github.com/baderlab/saber/stargazers)
[![GitHub-Forks](https://img.shields.io/github/forks/baderlab/saber.svg?colorB=blue&logo=github&logoColor=white)](https://github.com/BaderLab/saber/network/members)
[![GitHub-Commits](https://img.shields.io/github/commit-activity/y/baderlab/saber.svg?logo=git&logoColor=white)](https://github.com/baderlab/saber/graphs/commit-activity)
[![GitHub-Updated](https://img.shields.io/github/last-commit/baderlab/saber.svg?colorB=blue&logo=github)](https://github.com/baderlab/saber/pulse)

Pull and install straight from GitHub

```sh
(saber) $ pip install git+https://github.com/BaderLab/saber.git
```

or install by cloning the repository

```sh
(saber) $ git clone https://github.com/BaderLab/saber.git
(saber) $ cd saber
```

and then using either `pip`

```sh
(saber) $ pip install -e .
```
or `setuptools`

```sh
(saber) $ python setup.py install
```

See the [documentation](https://baderlab.github.io/saber/installation/) for more detailed installation instructions.

## Quickstart

If your goal is to use Saber to annotate biomedical text, then you can either use the [web-service](#web-service) or a [pre-trained model](#pre-trained-models). If you simply want to check Saber out, without installing anything locally, try the [Google Colaboratory](#google-colaboratory) notebook.

### Google Colaboratory

The fastest way to check out Saber is by following along with the Google Colaboratory notebook ([![Colab](https://img.shields.io/badge/launch-Google%20Colab-orange.svg)](https://colab.research.google.com/drive/1WD7oruVuTo6p_908MQWXRBdLF3Vw2MPo)). In order to be able to run the cells, select "Open in Playground" or, alternatively, save a copy to your own Google Drive account (File > Save a copy in Drive).

### Web-service

To use Saber as a **local** web-service, run

```
(saber) $ python -m saber.cli.app
```

or, if you prefer, you can pull & run the Saber image from **Docker Hub**

```sh
# Pull Saber image from Docker Hub
$ docker pull pathwaycommons/saber
# Run docker (use `-dt` instead of `-it` to run container in background)
$ docker run -it --rm -p 5000:5000 --name saber pathwaycommons/saber
```

There are currently two endpoints, `/annotate/text` and `/annotate/pmid`. Both expect a `POST` request with a JSON payload, e.g.,

```json
{
  "text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."
}
```

or

```json
{
  "pmid": 11835401
}
```

For example, running the web-service locally and using `cURL`

```sh
$ curl -X POST 'http://localhost:5000/annotate/text' \
--data '{"text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."}'
```

Documentation for the Saber web-service API can be found [here](https://baderlab.github.io/saber-api-docs/).

### Pre-trained models

First, import the `Saber` class. This is the interface to Saber

```python
from saber.saber import Saber
```

then create a `Saber` object

```python
saber = Saber()
```

and then load the model of our choice

```python
saber.load('PRGE')
```

To annotate text with the model, just call the `Saber.annotate()` method

```python
saber.annotate("The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.")
```
See the [documentation](https://baderlab.github.io/saber/quick_start/#pre-trained-models) for more details on using pre-trained models.

## Documentation

Documentation for the Saber package can be found [here](https://baderlab.github.io/saber/). The web-service API has its own documentation [here](https://baderlab.github.io/saber-api-docs/#introduction).
