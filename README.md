[![Build Status](https://travis-ci.org/BaderLab/Saber.svg?branch=master)](https://travis-ci.org/BaderLab/Saber)
[![Coverage Status](https://coveralls.io/repos/github/BaderLab/Saber/badge.svg?branch=master)](https://coveralls.io/github/BaderLab/Saber?branch=master)
[![Documentation Status](https://readthedocs.org/projects/saber-baderlab/badge/?version=latest)](http://saber-baderlab.readthedocs.io/en/latest/?badge=latest)
[![spaCy](https://img.shields.io/badge/spaCy-v2-09a3d5.svg)](https://spacy.io)
[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# Saber

**Saber** (**S**equence **A**nnotator for **B**iomedical **E**ntities and **R**elations) is a deep-learning based tool for **information extraction** in the biomedical domain.

### Requirements

Requires `Python >= 3.5`.

### Quickstart

First, clone the repository:

```
$ git clone https://github.com/BaderLab/Saber.git
```

When using pip it is generally recommended to install packages in a virtual environment to avoid modifying system state.

#### Using virtualenv or venv

To create the virtual environment.

Using [virtualenv](https://virtualenv.pypa.io/en/stable/):

```
$ virtualenv /path/to/new/venv
```

Using [venv](https://docs.python.org/3/library/venv.html):

```
$ python3 -m venv /path/to/new/venv
```

Next, you need to activate the environment.

On Windows (cmd.exe), run:

```
$ \path\to\new\venv\Scripts\activate.bat
```

On Unix or MacOS (bash/zsh), run:

```
$ source /path/to/new/venv/bin/activate
```

#### Using Conda

If you use Conda/Miniconda, you can create an environment by running:


```
$ conda create -n myenv python=3
```

To activate the environment:

```
$ source activate myenv
```

#### Installing requirements

With your **virtual environment activated**, install all requirements:

```
$ pip install -r requirements.txt
```

### Usage

You can interact with Saber as a command line tool, web-service or via the Juypter notebooks. If you created a virtual environment, **remember to activate it first**.

#### Command line tool

All hyper-parameters are specified in a configuration file. The configuration file can be specified when running Saber:

```
$ python main.py --config_filepath path/to/config.ini
```

If not specified, the default configuration file at `saber/config.ini` is used.

> Note: At this time, the command-line tool simply trains the model.

#### Web-service

To run Saber as a web-service, `cd` into the directory `saber` and run:

```
$ python app.py
```

To build Saber with Docker from the project root directory:

```
docker build -t saber .
```

To run: `docker run --rm -p 5000:5000 --name saber1 -dt saber` (use `-it` instead of `-dt` to try it interactively)


There are currently two endpoints, `/annotate/text` and `/annotate/pmid`. Both expect a POST request with a `json` payload, e.g.:

```
{
  "text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."
}
```

Or:

```
{
  "pmid": 11835401
}
```

For example, running the web-service locally and using `cURL`:

```
curl -XPOST --data '{"text": "The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53."}' 'http://localhost:5000/annotate/text'
```

> Currently, the pre-trained 'PRGE' model in the `pretrained_models/` folder powers the web-service.

#### Juypter notebooks

First, with your virtual environment activated run:

```
$ pip install jupyter lab
```

> Note, you only need to install this once!

Then `cd` into `saber` and run:

```
jupyter lab
```

Check out the `lightning_tour.ipynb` notebook for an overview.

### Resources

#### Datasets

Corpora are collected in the `datasets` folder for convenience. Many of the corpora in the IOB and IOBES tag format were originally collected by [Crichton _et al_., 2017](https://doi.org/10.1186/s12859-017-1776-8), [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

#### Word embeddings

You can provide your own pre-trained word embeddings with the `token_pretrained_embedding_filepath` argument (either at the command line or in the configuration file.) [Pyysalo _et al_. 2013](https://pdfs.semanticscholar.org/e2f2/8568031e1902d4f8ee818261f0f2c20de6dd.pdf) provide word embeddings that work quite well in the biomedical domain, which can be downloaded [here](http://bio.nlplab.org).
