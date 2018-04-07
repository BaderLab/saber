[![Coverage Status](https://coveralls.io/repos/github/BaderLab/Saber/badge.svg?branch=master)](https://coveralls.io/github/BaderLab/Saber?branch=master)
[![Build Status](https://travis-ci.org/BaderLab/Saber.svg?branch=master)](https://travis-ci.org/BaderLab/Saber)

# Saber

**Saber** (**S**equence **A**nnotator for **B**iomedical **E**ntities and **R**elations) is a deep-learning based tool for **information extraction** in the biomedical domain.

### Requirements

Requires `Python >= 3.5`. Requirements can be installed by calling:
```
$ pip install -r requirements.txt
```

It is recommended that you create a virtual environment first. See [here](https://docs.python.org/3/tutorial/venv.html).

### Usage

All hyper-parameters are specified in a configuration file. The configuration file can be specified when running __Saber__:

```
$ python main.py --config_filepath path/to/config.ini
```

If not specified, the default config file (`saber/config.ini`) is loaded.

There is also a **jupyter notebook** for interacting with Saber (`saber/Saber.ipynb`).
### Resources

#### Datasets

Corpora are collected in the **dataset** folder for convenience. Many of the corpora in the IOB and IOBES tag format were originally collected by [Crichton _et al_., 2017](https://doi.org/10.1186/s12859-017-1776-8), [here](https://github.com/cambridgeltl/MTL-Bioinformatics-2016).

#### Word embeddings

You can provide your own pre-trained word embeddings with the `token_pretrained_embedding_filepath` argument (either at the command line or in the configuration file.) [Pyysalo _et al_. 2013](https://pdfs.semanticscholar.org/e2f2/8568031e1902d4f8ee818261f0f2c20de6dd.pdf) provide word embeddings that work quite well in the biomedical domain, which can be downloaded [here](http://bio.nlplab.org).
