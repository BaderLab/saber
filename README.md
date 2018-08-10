<h1 align="center">
  <br>
  Saber
  <br>
</h1>

<p align="center"><b>Saber</b> (<b>S</b>equence <b>A</b>nnotator for <b>B</b>iomedical <b>E</b>ntities and <b>R</b>elations) is a deep-learning based tool for <b>information extraction</b> in the biomedical domain.
</p>

<p align="center">
  <a href="https://travis-ci.org/BaderLab/Saber">
    <img src="https://travis-ci.org/BaderLab/Saber.svg?branch=master"
         alt="Travis CI">
  </a>
  <a href="https://www.codacy.com/app/JohnGiorgi/Saber?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BaderLab/Saber&amp;utm_campaign=Badge_Grade">
    <img src="https://api.codacy.com/project/badge/Grade/d122e87152d84f959ee6d97b71d616cb" alt='Codacy Status'/>
  </a>
  <a href='https://coveralls.io/github/BaderLab/Saber?branch=master'>
    <img src='https://coveralls.io/repos/github/BaderLab/Saber/badge.svg?branch=master' alt='Coverage Status'/>
  </a>
  <a href='https://baderlab-saber.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/baderlab-saber/badge/?version=latest' alt='Documentation Status'/>
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
  <a href="#resources">Resources</a>
</p>

<p align="center">
  <img src="img/saber_main_img.png" alt="Size Limit example">
</p>

## Installation

**Disclaimer: Currently a pre-alpha, work in progress!**

To clone and run this application, you will need `python>=3.6`. If not already installed, `python>=3.6` can be installed via:

 - the [official installer](https://www.python.org/downloads/)
 - [Homebrew](https://brew.sh), on MacOS (`brew install python3`)
 - [Miniconda3](https://conda.io/miniconda.html) / [Anaconda3](https://www.anaconda.com/download/)

> In some cases, you may need to specify `python3` at the command line (not `python`) to ensure you use the installed `python >= 3.6` and not your systems default `python` installation. Use `python --version` to check.

From your command line:

```bash
# Clone this repository
$ git clone https://github.com/BaderLab/Saber.git

# Go into the repository
$ cd Saber

# OPTIONAL: activate your virtual environment (see below for help)
$ source activate saber

# Install dependencies
(saber) $ pip install -r requirements.txt
```

### Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`:

#### Using virtualenv or venv

Using [virtualenv](https://virtualenv.pypa.io/en/stable/):

```bash
$ virtualenv /path/to/new/venv/saber
```

Using [venv](https://docs.python.org/3/library/venv.html):

```bash
$ python3 -m venv /path/to/new/venv/saber
```

Next, you need to activate the environment.

```bash
$ source /path/to/new/venv/saber/bin/activate

# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

#### Using Conda

If you use [Conda](https://conda.io/docs/) / [Miniconda](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), you can create an environment named `saber` by running:

```bash
$ conda create -n saber python=3
```

To activate the environment:

```bash
$ source activate saber

# Again, your command prompt should change to indicate that the environment is active
(saber) $
```

> Note: you do not need to name the environment `saber`.

### Quickstart

You can interact with Saber as a web-service, command line tool, module, or via the Juypter notebooks. If you created a virtual environment, **remember to activate it first**.

> Note: the following examples assume you are in the project directory `Saber/saber`.

#### Web-service

To use Saber as a **local** web-service, run:

``` bash
(saber) $ python -m saber.app
```

To build & run Saber with __Docker__:

``` bash
# Build docker
(saber) $ docker build -t saber .

# Run docker (use `-it` instead of `-dt` to try it interactively)
(saber) $ docker run --rm -p 5000:5000 --name saber1 -dt saber
```

There are currently two endpoints, `/annotate/text` and `/annotate/pmid`. Both expect a `POST` request with a `json` payload, e.g.:

```json
{
  "text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."
}
```

Or:

```json
{
  "pmid": 11835401
}
```

For example, running the web-service locally and using `cURL`:

```bash
curl -X POST 'http://localhost:5000/annotate/text' \
--data '{"text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."}'
```

Full documentation for the Saber API can be found [here](https://baderlab.github.io/saber-api-docs/).

#### Command line tool

All hyper-parameters are specified in a configuration file. The configuration file can be specified when running Saber:

```bash
(saber) $ python -m saber.train --config_filepath path/to/config.ini
```

If not specified, the default configuration file at `saber/config.ini` is used.

Alternatively, you can supply arguments at the command line. Each command line argument has an identical name to those found in `saber/config.ini`. For example:

```bash
(saber) $ python -m saber.train --dataset_folder path/to/dataset --k_folds 10
```

Would overwrite the arguments for `dataset_folder` and `k_folds` found in the config file.

> Note: At this time, the command-line tool simply trains the model.

#### Python module

Saber exposes its functionality through the `SequenceProcessor` class. Here is a simple example where we load a pretrained model and use it to annotate raw text for protein and gene entities.

```python
from saber.sequence_processor import SequenceProcessor

# Create a SequenceProcessor object, which coordinates
# training/prediction/loading of models and datasets
sp = SequenceProcessor()

# Load the protein and gene entity model
sp.load('path/to/pretrained_models/PRGE')

# Text to annotate
raw_text = 'The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.'

# Perform prediction on raw text, get resulting annotation
annotation = sp.annotate(raw_text)
```

#### Juypter notebooks

First, install [Jupyter lab](https://github.com/jupyterlab/jupyterlab) (make sure to activate your virtual environment first if you created one):

```bash
# If you use pip, you can install it as
(saber) $ pip install jupyterlab

# If you use conda, you can install as
(saber) $ conda install -c conda-forge jupyterlab
```

> Note, you only need to install this once!

This is a temporary work-around, but you must also `pip install .` for the notebooks to work:

```
(saber) $ pip install .
```

Then `cd` into `saber` and run:

```
(saber) $ jupyter lab
```

Check out the `lightning_tour.ipynb` notebook for an overview.

## Resources

### Datasets

Corpora are collected in [here](https://github.com/BaderLab/Biomedical-Corpora) for convenience. Many of the corpora in the BIO and IOBES tag format were originally collected by [Crichton _et al_., 2017](https://doi.org/10.1186/s12859-017-1776-8), [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

### Word embeddings

When training new models, you can (and should) provide your own pre-trained word embeddings with the `token_pretrained_embedding_filepath` argument (either at the command line or in the configuration file). [Pyysalo _et al_. 2013](https://pdfs.semanticscholar.org/e2f2/8568031e1902d4f8ee818261f0f2c20de6dd.pdf) provide word embeddings that work quite well in the biomedical domain, which can be downloaded [here](http://bio.nlplab.org).

Once downloaded, you will need to convert them from `.bin` to `.txt` format:

```bash
(saber) $ cd saber
(saber) $ pip install gensim
(saber) $ python
>> from generic_utils import bin_to_txt
>> path_to_embeddings = '/path/to/wikipedia-pubmed-and-PMC-w2v.bin'
>> bin_to_txt('wikipedia-pubmed-and-PMC-w2v.bin', output_dir='path/to/word_embeddings')
```

> Note: you do not (and should not) have to download word embeddings if you only plan on using Saber's pretrained models.

## Running tests

Sabers test suite can be found in `Saber/saber/tests`. In order to run the tests, you'll usually want to clone the repository locally. Make sure to install all required development dependencies defined in the ``requirements.txt`` (see [Installation](#Installation) for more help). Additionally, you will
need to install ``pytest``:

```bash
(saber) $ pip install pytest
```

To run the tests:

```bash
(saber) $ cd path/to/Saber
(saber) $ py.test saber
```

Alternatively, you can find out where Saber is installed and run ``pytest`` on
that directory:

```bash
# Find out where Saber is installed
$ python -c "import os; import saber; print(os.path.dirname(saber.__file__))"
# Run tests on that installation directory
(saber) $ python -m pytest <Saber-directory>
```
