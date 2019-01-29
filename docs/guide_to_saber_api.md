# Guide to the Saber API

You can interact with Saber as a web-service (explained in [Quick Start: Web-service](https://baderlab.github.io/saber/quick_start/#web-service)), [command line tool](#command-line-tool), or as a [python package](#python-package). If you created a virtual environment, _remember to activate it first_.

### Command line tool

Currently, the command line tool simply trains the model. To use it, call

```
(saber) $ python -m saber.cli.train
```

along with any command line arguments. For example, to train the model on the [NCBI Disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) corpus

```
(saber) $ python -m saber.cli.train --dataset_folder NCBI_Disease_BIO
```

!!! tip
    See [Resources: Datasets](https://baderlab.github.io/saber/resources/#datasets) for help preparing datasets and word embeddings for training.

Run `python -m saber.cli.train --help` to see all possible arguments.

Of course, supplying arguments at the command line can quickly become cumbersome. Saber also allows you to provide a configuration file, which can be specified like so

```
(saber) $ python -m saber.cli.train --config_filepath path/to/config.ini
```

Copy the contents of the [default config file](https://github.com/BaderLab/saber/blob/master/saber/config.ini) to a new `*.ini` file in order to get started.

!!! note
    Arguments supplied at the command line overwrite those found in the configuration file, e.g.,

    ```
    (saber) $ python -m saber.cli.train --dataset_folder path/to/dataset --k_folds 10
    ```

    would overwrite the arguments for `dataset_folder` and `k_folds` found in the configuration file.

### Python package

You can also import Saber and interact with it as a python package. Saber exposes its functionality through the `Saber` class. Here is just about everything Saber does in one script:

```python
from saber.saber import Saber

# First, create a Saber object, which exposes Sabers functionality
saber = Saber()

# Load a dataset and create a model (provide a list of datasets to use multi-task learning!)
saber.load_dataset('path/to/datasets/GENIA')
saber.build(model_name='MT-LSTM-CRF')

# Train and save a model
saber.train()
saber.save('pretrained_models/GENIA')

# Load a model
del saber
saber = Saber()
saber.load('pretrained_models/GENIA')

# Perform prediction on raw text, get resulting annotation
raw_text = 'The phosphorylation of Hdm2 by MK2 promotes the ubiquitination of p53.'
annotation = saber.annotate(raw_text)

# Use transfer learning to continue training on a new dataset
saber.load_dataset('path/to/datasets/CRAFT')
saber.train()
```

#### Transfer learning

Transfer learning is as easy as training, saving, loading, and then continuing training of a model. Here is an example

```python
# Create and train a model on GENIA corpus
saber = Saber()
saber.load_dataset('path/to/datasets/GENIA')
saber.build(model_name='MT-LSTM-CRF')
saber.train()
saber.save('pretrained_models/GENIA')

# Load that model
del saber
saber = Saber()
saber.load('pretrained_models/GENIA')

# Use transfer learning to continue training on a new dataset
saber.load_dataset('path/to/datasets/CRAFT')
saber.train()
```

!!! note
    This is currently only supported by the `mt-lstm-crf` model.

#### Multi-task learning

Multi-task learning is as easy as specifying multiple dataset paths, either in the `config` file, at the command line via the flag `--dataset_folder`, or as an argument to `load_dataset()`. The number of datasets is arbitrary.

Here is an example using the last method

```python
saber = Saber()

# Simply pass multiple dataset paths as a list to load_dataset to use multi-task learning.
saber.load_dataset(['path/to/datasets/NCBI_Disease', 'path/to/datasets/Linnaeus'])

saber.build(model_name='MT-LSTM-CRF')
saber.train()
```

!!! note
    This is currently only supported by the `mt-lstm-crf` model.

#### Training on GPUs

Saber will automatically train on as many GPUs as are available. In order for this to work, you must have [CUDA](https://developer.nvidia.com/cuda-downloads) and, optionally, [CudDNN](https://developer.nvidia.com/cudnn) installed. If you are using conda to manage your environment, then these are installed for you when you call

```
(saber) $ conda install tensorflow-gpu
```

Otherwise, install them yourself and use `pip` to install `tensorflow-gpu`

```
(saber) $ pip install tensorflow-gpu
```

To control which GPUs Saber trains on, you can use the `CUDA_VISIBLE_DEVICES` environment variable, e.g.,

```
# To train exclusively on CPU
(saber) $ CUDA_VISIBLE_DEVICES="" python -m saber.cli.train

# To train on 1 GPU with ID=0
(saber) $ CUDA_VISIBLE_DEVICES="0" python -m saber.cli.train

# To train on 2 GPUs with IDs=0,2
(saber) $ CUDA_VISIBLE_DEVICES="0,2" python -m saber.cli.train
```

!!! tip
    You can get information about your NVIDIA GPUs by typing `nvidia-smi` at the command line (assuming the GPUs are setup properly and the nvidia driver is installed).

#### Saving and loading models

In the following sections we introduce the saving and loading of models.

##### Saving a model

Assuming the model has already been created (see above), we can easily save our model like so

```python
save_dir = 'path/to/pretrained_models/mymodel'
saber.save(save_dir)
```

##### Loading a model

Lets illustrate loading a model with a new `Saber` object

```python
# Delete our previous Saber object (if it exists)
del saber
# Create a new Saber object
saber = Saber()
# Load a previous model
saber.load(path_to_saved_model)
```
