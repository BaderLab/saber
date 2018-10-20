<h1 align="center">
  <br>
  Saber
  <br>
</h1>

<p align="center"><b>Saber</b> (<b>S</b>equence <b>A</b>nnotator for <b>B</b>iomedical <b>E</b>ntities and <b>R</b>elations) is a deep-learning based tool for <b>information extraction</b> in the biomedical domain.
</p>

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
  <a href='https://opensource.org/licenses/MIT'>
    <img src='https://img.shields.io/badge/License-MIT-blue.svg' alt='License'/>
  </a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#documentation">Documentation</a>
</p>

<p align="center">
  <img src="img/saber_main_img.png" alt="Size Limit example">
</p>

## Installation

**Disclaimer: Currently a pre-alpha, work in progress!**

To install Saber, you will need `python==3.6`. If not already installed, `python==3.6` can be installed via

 - the [official installer](https://www.python.org/downloads/)
 - [Homebrew](https://brew.sh), on MacOS (`brew install python3`)
 - [Miniconda3](https://conda.io/miniconda.html) / [Anaconda3](https://www.anaconda.com/download/)

> Use `python --version` at the command line to make sure installation was successful. Note: you may need to use `python3` (not just `python`) at the command line depending on your install method.

(OPTIONAL) Activate your virtual environment (see [below](#optional-creating-and-activating-virtual-environments) for help)

```bash
$ conda activate saber
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

then install Saber right from this repository with `pip`

```bash
(saber) $ pip install git+https://github.com/BaderLab/saber.git
```

or by cloning the repository and then using `pip` to install the package

```bash
(saber) $ git clone https://github.com/BaderLab/saber.git
(saber) $ cd saber
(saber) $ pip install .
```

> You can also install Saber by cloning this repository and running `python setup.py install`

Finally, you must also `pip` install the required [Spacy](https://spacy.io) model and the [keras-contrib](https://github.com/keras-team/keras-contrib) repositories

```bash
# keras-contrib
(saber) $ pip install git+https://www.github.com/keras-team/keras-contrib.git
# NeuralCoref medium model built on top of Spacy, this might take a few minutes to download!
(saber) $ pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz
```

### (OPTIONAL) Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`

#### Using virtualenv or venv

Using [virtualenv](https://virtualenv.pypa.io/en/stable/)

```bash
$ virtualenv --python=python3 /path/to/new/venv/saber
```

Using [venv](https://docs.python.org/3/library/venv.html)

```bash
$ python3 -m venv /path/to/new/venv/saber
```

Next, you need to activate the environment

```bash
$ source /path/to/new/venv/saber/bin/activate

# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

#### Using Conda

If you use [Conda](https://conda.io/docs/) / [Miniconda](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), you can create an environment named `saber` by running

```bash
$ conda create -n saber python=3.6
```

To activate the environment:

```bash
$ conda activate saber

# Again, your command prompt should change to indicate that the environment is active
(saber) $
```

> Note: you do not need to name the environment `saber`.

## Quickstart

If your goal is simply to use Saber to annotate biomedical text, then you can either use the [web-service](#web-service) or a [pre-trained model](#pre-trained-models).

### Web-service

To use Saber as a **local** web-service, run

```bash
(saber) $ python -m saber.cli.app
```

or, if you prefer, you can pull & run the Saber image from **Docker Hub**

```bash
# Pull Saber image from Docker Hub
$ docker pull pathwaycommons/saber
# Run docker (use `-dt` instead of `-it` to run container in background)
$ docker run -it --rm -p 5000:5000 --name saber pathwaycommons/saber
```

There are currently two endpoints, `/annotate/text` and `/annotate/pmid`. Both expect a `POST` request with a JSON payload, e.g.

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

```bash
$ curl -X POST 'http://localhost:5000/annotate/text' \
--data '{"text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."}'
```

Documentation for the Saber web-service API can be found [here](https://baderlab.github.io/saber-api-docs/). We hope to provide a live version of the web-service soon!

### Pre-trained models

First, import the `Saber` class. This is the interface to Saber

```python
from saber.saber import Saber
```

To load a pre-trained model, we first create a `Saber` object

```python
saber = Saber()
```

and then load the model of our choice

```python
saber.load('PRGE')
```

You can see all the pre-trained models in the [web-service API docs](https://baderlab.github.io/saber-api-docs/) or, alternatively, by running the following line of code

```python
from saber.constants import ENTITIES; print(list(ENTITIES.keys()))
```

To annotate text with the model, just call the `annotate()` method

```python
saber.annotate("The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.")
```

See the [documentation](https://baderlab.github.io/saber/quick_start/) for more details.

## Documentation

Documentation for the Saber API can be found [here](https://baderlab.github.io/saber/). The web-service API has its own documentation [here](https://baderlab.github.io/saber-api-docs/#introduction). Finally, we provide a [jupyter notebook](notebooks/lightning_tour.ipynb) which introduces the main ways of using Saber. See [here](https://baderlab.github.io/saber/guide_to_saber_api/#juypter-notebooks) for help setting up [JupyterLab](https://github.com/jupyterlab/jupyterlab).
