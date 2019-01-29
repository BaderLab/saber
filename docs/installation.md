# Installation

To install Saber, you will need `python3.6`. If not already installed, `python3` can be installed via

- The [official installer](https://www.python.org/downloads/)
- [Homebrew](https://brew.sh), on MacOS (`brew install python3`)
- [Miniconda3](https://conda.io/miniconda.html) / [Anaconda3](https://www.anaconda.com/download/)

!!! note
    Run `python --version` at the command line to make sure installation was successful. You may need to type `python3` (not just `python`) depending on your install method.

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

!!! note
    See [Running tests](#running-tests) for a way to verify your installation.

## (OPTIONAL) Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`

### Using virtualenv or venv

Using [virtualenv](https://virtualenv.pypa.io/en/stable/)

```
$ virtualenv --python=python3 /path/to/new/venv/saber
```

Using [venv](https://docs.python.org/3/library/venv.html)

```
$ python3 -m venv /path/to/new/venv/saber
```

Next, you need to activate the environment.

```
$ source /path/to/new/venv/saber/bin/activate
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

### Using Conda

If you use [Conda](https://conda.io/docs/) / [Miniconda](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), you can create an environment named `saber` by running

```
$ conda create -n saber python=3.6
```

To activate the environment

```
$ conda activate saber
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

!!! note
    You do not _need_ to name the environment `saber`.

## Running tests

Sabers test suite can be found in `saber/tests`. If Saber is already installed, you can run `pytest` on the installation directory

```
# Install pytest
(saber) $ pip install pytest
# Find out where Saber is installed
(saber) $ INSTALL_DIR=$(python -c "import os; import saber; print(os.path.dirname(saber.__file__))")
# Run tests on that installation directory
(saber) $ python -m pytest $INSTALL_DIR
```

Alternatively, to clone Saber, install it, and run the test suite all in one go

```
(saber) $ git clone https://github.com/BaderLab/saber.git
(saber) $ cd saber
(saber) $ python setup.py test
```
