<p align="center">
  <img src="img/saber_logo.png", style="height:150px">
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

To install Saber, you will need `python>=3.5`. If not already installed, `python>=3.5` can be installed via

 - The [official installer](https://www.python.org/downloads/)
 - [Homebrew](https://brew.sh), on MacOS (`brew install python3`)
 - [Miniconda3](https://conda.io/miniconda.html) / [Anaconda3](https://www.anaconda.com/download/)

(OPTIONAL) Activate your virtual environment (see [below](#optional-creating-and-activating-virtual-environments) for help)

```sh
$ conda activate saber
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

then install Saber

```sh
(saber) $ pip install saber
```

To get the latest development version of Saber, install it right from this repository with `pip`

```sh
(saber) $ pip install https://github.com/BaderLab/saber.git
```

or by cloning the repository and then using `pip` to install the package

```sh
(saber) $ git clone https://github.com/BaderLab/saber.git
(saber) $ cd saber
(saber) $ pip install .
```

For now, you will need to install the required [Spacy](https://spacy.io) model and the [keras-contrib](https://github.com/keras-team/keras-contrib) repository (even if you installed with `pip install saber`)

```sh
# keras-contrib
(saber) $ pip install git+https://www.github.com/keras-team/keras-contrib.git
# NeuralCoref medium model built on top of Spacy, this might take a few minutes to download!
(saber) $ pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz
```

### (OPTIONAL) Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`

#### Using virtualenv or venv

Using [virtualenv](https://virtualenv.pypa.io/en/stable/)

```sh
$ virtualenv --python=python3 /path/to/new/venv/saber
```

Using [venv](https://docs.python.org/3/library/venv.html)

```sh
$ python3 -m venv /path/to/new/venv/saber
```

Next, you need to activate the environment

```sh
$ source /path/to/new/venv/saber/bin/activate
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

#### Using Conda

If you use [Conda](https://conda.io/docs/), you can create an environment named `saber` by running

```sh
$ conda create -n saber python=3.6
```

then activate the environment with

```sh
$ conda activate saber
# Again, your command prompt should change to indicate that the environment is active
(saber) $
```

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

To load a pre-trained model, first create a `Saber` object

```python
saber = Saber()
```

and then load the model of our choice

```python
saber.load('PRGE')
```

You can see all the pre-trained models in the [web-service API docs](https://baderlab.github.io/saber-api-docs/) or, the [saber/pretrained_models](saber/pretrained_models) folder in this repository, or by running the following line of code

```python
from saber.constants import ENTITIES; print(list(ENTITIES.keys()))
```

To annotate text with the model, just call the `Saber.annotate()` method

```python
saber.annotate("The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.")
```
See the [documentation](https://baderlab.github.io/saber/quick_start/) for more details.

## Documentation

Documentation for the Saber API can be found [here](https://baderlab.github.io/saber/). The web-service API has its own documentation [here](https://baderlab.github.io/saber-api-docs/#introduction). Finally, we provide a [jupyter notebook](notebooks/lightning_tour.ipynb) which introduces the main ways of using Saber. See [here](https://baderlab.github.io/saber/guide_to_saber_api/#juypter-notebooks) for help setting up [JupyterLab](https://github.com/jupyterlab/jupyterlab).
