# Installation

To install Saber, you will need `python>=3.5`. If not already installed, `python>=3.5` can be installed via

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

!!! note
    See [Running tests](#running-tests) for a way to verify your installation.

### (OPTIONAL) Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`

#### Using virtualenv or venv

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

#### Using Conda

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
