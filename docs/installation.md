# Installation

To install Saber, you will need `python==3.6`. If not already installed, `python==3.6` can be installed via:

 - the [official installer](https://www.python.org/downloads/)
 - [Homebrew](https://brew.sh), on MacOS (`brew install python3`)
 - [Miniconda3](https://conda.io/miniconda.html) / [Anaconda3](https://www.anaconda.com/download/)

> Use `python --version` at the command line to make sure installation was successful. Note: you may need to use `python3` (not just `python`) at the command line depending on your install method.

(OPTIONAL) First, activate your virtual environment (see [below](#optional-creating-and-activating-virtual-environments) for help):

```
$ source activate saber
(saber) $
```

Then, install Saber right from the repository with `pip`

```
(saber) $ pip install git+https://github.com/BaderLab/saber.git
```

or by cloning the repository and then using `pip` to install the package

```
(saber) $ git clone https://github.com/BaderLab/saber.git
(saber) $ cd saber
(saber) $ pip install .
```

> You can also install Saber by cloning this repository and running `python setup.py install` from within the repository.

Finally, you must also `pip` install the required [Spacy](https://spacy.io) model and the [keras-contrib](https://github.com/keras-team/keras-contrib) repositories

```
# keras-contrib
(saber) $ pip install git+https://www.github.com/keras-team/keras-contrib.git
# NeuralCoref Large model built on top of Spacy, this might take a while to download!
(saber) $ pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_lg-3.0.0/en_coref_lg-3.0.0.tar.gz
```

> See [Running tests](#running-tests) for a way to verify your installation.


### (OPTIONAL) Creating and activating virtual environments

When using `pip` it is generally recommended to install packages in a virtual environment to avoid modifying system state. To create a virtual environment named `saber`:

#### Using virtualenv or venv

Using [virtualenv](https://virtualenv.pypa.io/en/stable/):

```
$ virtualenv /path/to/new/venv/saber
```

Using [venv](https://docs.python.org/3/library/venv.html):

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

If you use [Conda](https://conda.io/docs/) / [Miniconda](https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh), you can create an environment named `saber` by running:

```
$ conda create -n saber python=3.6
```

To activate the environment:

```
$ source activate saber
# Notice your command prompt has changed to indicate that the environment is active
(saber) $
```

> Note: you do not need to name the environment `saber`.

## Running tests

Sabers test suite can be found in `saber/tests`. In order to run the tests, you'll usually want to clone the repository locally. Make sure to install all required development dependencies defined in ``requirements.txt``. Additionally, you will need to install ``pytest``:

```
(saber) $ pip install pytest
```

To run the tests:

```
(saber) $ cd path/to/saber
(saber) $ py.test saber
```

Alternatively, you can find out where Saber is installed and run `pytest` on that directory:

```
# Find out where Saber is installed
(saber) $ python -c "import os; import saber; print(os.path.dirname(saber.__file__))"
# Run tests on that installation directory
(saber) $ python -m pytest <Saber-directory>
```
