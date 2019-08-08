# Installation

Make sure you have python 3.6 or newer. You can then install Saber using pip. If not already installed, python 3 can be installed via

- The [official installer](https://www.python.org/downloads/)
- [Homebrew](https://brew.sh), on MacOS (`brew install python3`)
- [Miniconda3](https://conda.io/miniconda.html) / [Anaconda3](https://www.anaconda.com/download/)

!!! note
    Run `python --version` at the command line to make sure installation was successful. You may need to type `python3` (not just `python`) depending on your installation method.

(OPTIONAL) Activate your virtual environment (see [below](#optional-creating-and-activating-virtual-environments) for help)

```sh
$ conda activate saber
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

## Latest PyPI stable release

[![PyPI-Status](https://img.shields.io/pypi/v/saber.svg?colorB=blue&style=flat-square)](https://pypi.org/project/saber/)
[![PyPI-Downloads](https://img.shields.io/pypi/dm/saber.svg?colorB=blue&style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/saber)
[![Libraries-Dependents](https://img.shields.io/librariesio/dependent-repos/pypi/saber.svg?colorB=blue&style=flat-square&logo=koding&logoColor=white)](https://github.com/baderlab/saber/network/dependents)

```sh
(saber) $ pip install saber
```

!!! error
    The install from PyPI is currently broken, please install using the instructions below.

## Latest development release on GitHub

[![GitHub-Status](https://img.shields.io/github/tag-date/baderlab/saber.svg?logo=github&style=flat-square)](https://github.com/baderlab/saber/releases)
[![GitHub-Stars](https://img.shields.io/github/stars/baderlab/saber.svg?logo=github&label=stars&style=flat-square)](https://github.com/baderlab/saber/stargazers)
[![GitHub-Forks](https://img.shields.io/github/forks/baderlab/saber.svg?colorB=blue&logo=github&logoColor=white&style=flat-square)](https://github.com/BaderLab/saber/network/members)
[![GitHub-Commits](https://img.shields.io/github/commit-activity/y/baderlab/saber.svg?logo=git&logoColor=white&style=flat-square)](https://github.com/baderlab/saber/graphs/commit-activity)
[![GitHub-Updated](https://img.shields.io/github/last-commit/baderlab/saber.svg?colorB=blue&logo=github&style=flat-square)](https://github.com/baderlab/saber/pulse)

Pull and install in the current directory:

```
(saber) $ pip install -e git+https://github.com/BaderLab/saber.git@master#egg=saber
```

## Other installation requirements

Regardless of installation method, you will need to additionally download a [SpaCy](https://spacy.io/usage) language model

```
(saber) $ python -m spacy download en_core_web_md
```

For GPU support, make sure to install PyTorch 1.0.0+ with CUDA support. See [here](https://pytorch.org/get-started/locally/) for instructions.

!!! note
    See [Running tests](#running-tests) for a way to verify your installation.

## (OPTIONAL) Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`

### Using virtualenv or venv

Decide where you want the environment to live, e.g.

```bash
$ ENV=~/saber
```

Then, create the environment using [virtualenv](https://virtualenv.pypa.io/en/stable/)

```bash
$ virtualenv --python=python3 $ENV
```

or [venv](https://docs.python.org/3/library/venv.html)

```bash
$ python3 -m venv $ENV
```

Finally, activate the environment

```bash
$ source $ENV/bin/activate
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

### Using Conda

If you use [Conda](https://conda.io/docs/) / [Miniconda](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), you can create an environment named `saber` by running

```bash
$ conda create -n saber python=3.7
```

To activate the environment

```bash
$ conda activate saber
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

## Running tests

Sabers test suite can be found in `saber/tests`. If Saber is already installed, you can run `pytest` on the installation directory

```bash
# Install pytest
(saber) $ pip install pytest
# Find out where Saber is installed
(saber) $ INSTALL_DIR=$(python -c "import os; import saber; print(os.path.dirname(saber.__file__))")
# Run tests on that installation directory
(saber) $ python -m pytest $INSTALL_DIR
```

Alternatively, to clone Saber, install it, and run the test suite all in one go

```bash
(saber) $ git clone https://github.com/BaderLab/saber.git
(saber) $ cd saber
(saber) $ python setup.py test
```
