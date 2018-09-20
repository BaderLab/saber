# Guide to the Saber API

You can interact with Saber as a web-service (explained in [Quick start](https://baderlab.github.io/saber/quick_start/)), command line tool, python package, or via the Juypter notebooks. If you created a virtual environment, _remember to activate it first_.

### Command line tool

Currently, the command line tool simply trains the model. To use it, call

```
(saber) $ python -m saber.train
```

along with any command line arguments. For example, to train the model on the [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) corpus

```
(saber) $ python -m saber.train --dataset_folder NCBI_disease_BIO
```

> See [Resources](https://baderlab.github.io/saber/resources/) for help preparing datasets for training.

Run `python -m saber.train -h` to see all possible arguments.

Of course, supplying arguments at the command line can quickly become cumbersome. Saber also allows you to specify a configuration file, which can be specified like so

```
(saber) $ python -m saber.train --config_filepath path/to/config.ini
```

Copy the contents of the [default config file](https://github.com/BaderLab/saber/blob/master/saber/config.ini) to a new `*.ini` file in order to get started.

Note that arguments supplied at the command line overwrite those found in the configuration file. For example

```
(saber) $ python -m saber.train --dataset_folder path/to/dataset --k_folds 10
```

would overwrite the arguments for `dataset_folder` and `k_folds` found in the configuration file.

### Python package

You can also import Saber and interact with it as a python package. Saber exposes its functionality through the `SequenceProcessor` class. Here is just about everything Saber does in one script:

```python
from saber.sequence_processor import SequenceProcessor

# First, create a SequenceProcessor object, which exposes Sabers functionality
sp = SequenceProcessor()

# Load a dataset and create a model (provide a list of datasets to use multi-task learning!)
sp.load_dataset('path/to/datasets/GENIA')
sp.create_model()

# Train and save a model
sp.fit()
sp.save('pretrained_models/GENIA')

# Load a model
del sp
sp = SequenceProcessor()
sp.load('pretrained_models/GENIA')

# Perform prediction on raw text, get resulting annotation
raw_text = 'The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.'
annotation = sp.annotate(raw_text)

# Use transfer learning to continue training on a new dataset
sp.load_dataset('path/to/datasets/CRAFT')
sp.fit()
```

#### Transfer learning

Transfer learning is as easy as training, saving, loading, and then continuing training of a model. Here is an example

```python
# Create and train a model on GENIA corpus
sp = SequenceProcessor()
sp.load_dataset('path/to/datasets/GENIA')
sp.create_model()
sp.fit()
sp.save('pretrained_models/GENIA')

# Load that model
del sp
sp = SequenceProcessor()
sp.load('pretrained_models/GENIA')

# Use transfer learning to continue training on a new dataset
sp.load_dataset('path/to/datasets/CRAFT')
sp.fit()
```

> Note that there is currently no way to easily do this with the command line interface, but I am working on it!

#### Multi-task learning

Multi-task learning is as easy as specifying multiple dataset paths, either in the `config` file, at the command line via the flag `--dataset_folder`, or as an argument to `load_dataset()`. The number of datasets is arbitrary.

Here is an example using the last method

```python
sp = SequenceProcessor()

# Simply pass multiple dataset paths as a list to load_dataset to use multi-task learning.
sp.load_dataset(['path/to/datasets/NCBI-Disease', 'path/to/datasets/Linnaeus'])

sp.create_model()
sp.fit()
```

#### Saving and loading models

In the following sections we introduce the saving and loading of models.

##### Saving a model

Assuming the model has already been created (see above), we can easily save our model like so

```python
path_to_saved_model = 'path/to/pretrained_models/mymodel'

sp.save(path_to_saved_model)
```

##### Loading a model

Lets illustrate loading a model with a new `SequenceProccesor` object

```python
# Delete our previous SequenceProccesor object (if it exists)
if 'sp' in locals(): del sp

# Create a new SequenceProccesor object
sp = SequenceProcessor()

# Load a previous model
sp.load(path_to_saved_model)
```

### Juypter notebooks

First, grab the notebook. Go [here](https://raw.githubusercontent.com/BaderLab/saber/master/notebooks/lightning_tour.ipynb), then press `command` / `control` + `s`, and save the notebook as `lightning_tour.ipynb` somewhere on your computer.


Next, install [JupyterLab](https://github.com/jupyterlab/jupyterlab) by following the instructions [here](https://github.com/jupyterlab/jupyterlab#installation). Once installed, run:

```
(saber) $ jupyter lab
```

A new window will open in your browser. Use it to search for `lightning_tour.ipynb` on your computer.

A couple of notes:

- If you activated a virtual environment, "`myenv`", make sure you see **Python [venv:myenv]** in the top right of the Jupyter notebook.
- If you are using conda, you need to run `conda install nb_conda` with your environment activated.
