"""Contains the BaseModel class, the parent class to all Keras models in Saber.
"""
import json
import logging
import os

import torch
from keras.models import Model
from torch import nn
from keras.models import model_from_json

from .. import constants

LOGGER = logging.getLogger(__name__)


class BaseModel():
    """Parent class of all deep learning models implemented by Saber.

    Attributes:
        config (Config): A Config object which contains a set of harmonized arguments provided in a
            *.ini file and, optionally, from the command line.
        datasets (list): A list containing one or more Dataset objects.
        embeddings (Embeddings): An object containing loaded word embeddings.
        models (list): A list of Keras or PyTorch models.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        self.config = config  # Hyperparameters and model details
        self.datasets = datasets  # Dataset(s) tied to this instance
        self.embeddings = embeddings  # Pre-trained word embeddings tied to this instance
        self.model = None  # Saber model tied to this instance

        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset_model(self):
        """Clears and rebuilds the model.

        Clear and rebuilds the model(s) at `self.models`. This is useful, for example, at the end
        of a cross-validation fold.
        """
        self.model = None
        self.specify()
        self.compile()


class BaseKerasModel(BaseModel):
    """Parent class of all Keras model classes implemented by Saber.

    config (Config): A Config object which contains a set of harmonized arguments provided in a
        *.ini file and, optionally, from the command line.
    datasets (list): A list containing one or more Dataset objects.
    embeddings (Embeddings): An object containing loaded word embeddings.
    models (list): A list of Keras models.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        super().__init__(config, datasets, embeddings, **kwargs)

        # attribute we can use to identify which framework / library model is written in
        self.framework = constants.KERAS

        # set Tensorflow logging level
        if self.config.verbose:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        else:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    def save(self, model_filepath, weights_filepath):
        """Save a Keras model to disk.

        Saves a Keras model to disk, by saving its architecture as a `.json` file at
        `model_filepath` and its weights as a `.hdf5` file at `model_filepath`.

        Args:
            model_filepath (str): Filepath to the models architecture (`.json` file).
            weights_filepath (str): Filepath to the models weights (`.hdf5` file).
            model (keras.Model): Optional, model to save to disk. Defaults to `self.model`.
        """
        with open(model_filepath, 'w') as f:
            model_json = self.model.to_json()
            json.dump(json.loads(model_json), f)
            self.model.save_weights(weights_filepath)

    def load(self, model_filepath, weights_filepath, custom_objects=None):
        """Load a Keras model from disk.

        Loads a Keras model from disk by loading its architecture from a `.json` file at
        `model_filepath` and its weights from a `.hdf5` file at `model_filepath`.

        Args:
            model_filepath (str): Filepath to the models architecture (`.json` file).
            weights_filepath (str): Filepath to the models weights (`.hdf5` file).
            custom_objects: Optional, dictionary mapping names (strings) to custom classes or
                functions to be considered during deserialization.

        Returns:
            The Keras `Model` object that was saved to disk.
        """
        with open(model_filepath) as f:
            model = model_from_json(f.read(), custom_objects=custom_objects)
            model.load_weights(weights_filepath)
            self.model = model

        return model

    def summary(self):
        """Prints a summary representation of the Keras model `self.model`.
        """
        self.model.summary()

    def prune_output_layers(self, indices):
        """Removes output layers with indicies not in `output_layer_indices` in `self.model`.

        Args:
            output_layer_indices (int or list): An integer index or list of inter indicies into the
                output layers of `self.model` to retain.

        Returns:
            `self.model`, where any output layers with indicies not in `output_layer_indices` have
            been removed.

        Raises:
            ValueError if `not isinstance(self.model.output, list)`.
        """
        if not isinstance(self.model.output, list):
            err_msg = (f'Tried to call `prune_output_layers()` for a Model object ({self.model})'
                       ' with a single output layer.')
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        n_outputs = len(self.model.output)
        n_layers = len(self.model.layers)

        # Allow user to supply int or list of ints
        indices = [indices] if not isinstance(indices, list) else indices

        last_hidden_layer = self.model.get_layer(index=-(n_outputs + 1))
        output_layers = [self.model.get_layer(index=n_layers - n_outputs + i)
                         for i in range(n_outputs)]

        outputs = [output_layers[i](last_hidden_layer.output) for i in indices]

        self.model = Model(self.model.input, outputs)

        return self.model


class BasePyTorchModel(BaseModel):
    """Parent class of all PyTorch model classes implemented by Saber.

    config (Config): A Config object which contains a set of harmonized arguments provided in a
        *.ini file and, optionally, from the command line.
    datasets (list): A list containing one or more Dataset objects.
    embeddings (Embeddings): An object containing loaded word embeddings.
    models (list): A list of PyTorch models.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        super().__init__(config, datasets, embeddings, **kwargs)

        # Attribute we can use to identify which framework / library model is written in
        self.framework = constants.PYTORCH

    def save(self, model_filepath, model_idx=-1):
        """Save a PyTorch model to disk.

        Saves a PyTorch model to disk, by saving its architecture and weights as a `.bin` file
        at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).
            model_idx (int): Index to model in `self.models` that will be saved. Defaults to -1.
        """
        torch.save(self.model.state_dict(), model_filepath)

    def compile(self):
        """Dummy function, does nothing.

        This is a dummy function which makes code simpler elsewhere in Saber. PyTorch models
        don't need to be compiled, but Keras models do. To avoid writing extra code, both
        Keras and PyTorch models implement a `compile` method.
        """
        pass

    def summary(self):
        """Prints a summary representation of the PyTorch model `self.model`.
        """
        pass

    def prune_output_layers(self, indices):
        """Removes output layers with indicies not in `output_layer_indices` in `self.model`.

        Args:
            output_layer_indices (int or list): An integer index or list of inter indicies into the
                output layers of `self.model` to retain.

        Returns:
            `self.model`, where any output layers with indicies not in `output_layer_indices` have
            been removed.

        Raises:
            ValueError if `len(self.model.classifier) < 2`
        """
        if len(self.model.classifier) < 2:
            err_msg = ('Tried to call `prune_output_layers()` for a nn.Module object'
                       f' ({self.model}) with a single output layer.')
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        # Allow user to supply int or list of ints
        indices = [indices] if not isinstance(indices, list) else indices

        self.model.classifier = nn.ModuleList(
            [self.model.classifier[i] for i in indices]
        )

        return self.model
