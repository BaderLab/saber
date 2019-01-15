"""Contains the BaseModel class, the parent class to all Keras models in Saber.
"""
import json
import logging

from keras import optimizers
from keras.models import model_from_json

import torch

LOGGER = logging.getLogger(__name__)

class BaseModel(object):
    """Parent class of all deep learning models implemented by Saber.

    config (Config): A Config object which contains a set of harmonized arguments provided in a
        *.ini file and, optionally, from the command line.
    datasets (list): A list containing one or more Dataset objects.
    embeddings (Embeddings): An object containing loaded word embeddings.
    models (list): A list of Keras or PyTorch models.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        self.config = config  # hyperparameters and model details
        self.datasets = datasets  # dataset(s) tied to this instance
        self.embeddings = embeddings  # pre-trained word embeddings tied to this instance
        self.models = []  # Keras / PyTorch model(s) tied to this instance

        for key, value in kwargs.items():
            setattr(self, key, value)

class BaseKerasModel(BaseModel):
    """Parent class of all Keras model classes implemented by Saber.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        super().__init__(config, datasets, embeddings, **kwargs)

    def save(self, weights_filepath, model_filepath, model_idx=0):
        """Save a Keras model to disk.

        Saves a Keras model to disk, by saving its architecture as a `.json` file at
        `model_filepath` and its weights as a `.hdf5` file at `model_filepath`.

        Args:
            weights_filepath (str): Filepath to the models weights (`.hdf5` file).
            model_filepath (str): Filepath to the models architecture (`.json` file).
            model_idx (int): Index to model in `self.models` that will be saved. Defaults to 0.
        """
        with open(model_filepath, 'w') as f:
            model_json = self.models[model].to_json()
            json.dump(json.loads(model_json), f, sort_keys=True, indent=4)
            self.models[model].save_weights(weights_filepath)

    def load(self, weights_filepath, model_filepath):
        """Load a Keras model from disk.

        Loads a Keras model from disk by loading its architecture from a `.json` file at
        `model_filepath` and its weights from a `.hdf5` file at `model_filepath`.

        Args:
            weights_filepath (str): Filepath to the models weights (`.hdf5` file).
            model_filepath (str): Filepath to the models architecture (`.json` file).
        """
        with open(model_filepath) as f:
            model = model_from_json(f.read())
            model.load_weights(weights_filepath)
            self.models.append(model)

    def _compile(self, model, loss_function, optimizer, lr=0.01, decay=0.0, clipnorm=0.0):
        """Compiles a model specified with Keras.

        See https://keras.io/optimizers/ for more info on each optimizer.

        Args:
            model (keras.Model): Keras model object to compile.
            loss_function: A loss function to be passed to `keras.Model.compile()`.
            optimizer (str): The optimizer to use during training.
            lr (float): Learning rate to use during training
            decay (float): Per epoch decay rate.
            clipnorm (float): Gradient normalization threshold.
        """
        # The parameters of these optimizers can be freely tuned.
        if optimizer == 'sgd':
            optimizer_ = optimizers.SGD(lr=lr, decay=decay, clipnorm=clipnorm)
        elif optimizer == 'adam':
            optimizer_ = optimizers.Adam(lr=lr, decay=decay, clipnorm=clipnorm)
        elif optimizer == 'adamax':
            optimizer_ = optimizers.Adamax(lr=lr, decay=decay, clipnorm=clipnorm)
        # It is recommended to leave the parameters of this optimizer at their
        # default values (except the learning rate, which can be freely tuned).
        # This optimizer is usually a good choice for recurrent neural networks
        elif optimizer == 'rmsprop':
            optimizer_ = optimizers.RMSprop(lr=lr, clipnorm=clipnorm)
        # It is recommended to leave the parameters of these optimizers at their
        # default values.
        elif optimizer == 'adagrad':
            optimizer_ = optimizers.Adagrad(clipnorm=clipnorm)
        elif optimizer == 'adadelta':
            optimizer_ = optimizers.Adadelta(clipnorm=clipnorm)
        elif optimizer == 'nadam':
            optimizer_ = optimizers.Nadam(clipnorm=clipnorm)
        else:
            err_msg = "Argument for `optimizer` is invalid, got: {}".format(optimizer)
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        model.compile(optimizer=optimizer_, loss=loss_function)

class BasePyTorchModel(BaseModel):
    """Parent class of all PyTorch model classes implemented by Saber.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        super().__init__(config, datasets, embeddings, **kwargs)

    def save(self, model_filepath, model_idx=0):
        """Save a PyTorch model to disk.

        Saves a PyTorch model to disk, by saving its architecture and weights as a `.bin` file
        at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).
            model_idx (int): Index to model in `self.models` that will be saved. Defaults to 0.
        """
        # TODO (James): Fill this in based on your stuff in the notebook
        # TODO (James): In the future, we would like to support MTL. So self.models is a list.
        # by default, this function should save the first model in that list
        ### YOUR CODE STARTS HERE ####
        # torch.save(self.models[model].state_dict(), model_filepath)
        ### YOUR CODE ENDS HERE ####
        pass

    def load(self, model_filepath):
        """Load a model from disk.

        Loads a PyTorch model from disk by loading its architecture and weights from a `.bin` file
        at `model_filepath`.

        Args:
            model_filepath (str): filepath to the models architecture (`.bin` file).
        """
        # TODO (James): Fill this in based on your stuff in the notebook
        # TODO (James): In the future, we would like to support MTL. So self.models is a list.
        # write the most generic, best practice way to load models here
        ### YOUR CODE STARTS HERE ####
        ### YOUR CODE ENDS HERE ####
        # self.models.append(model)
        pass
