"""Contains the BaseModel class, the parent class to all Keras models in Saber.
"""
import logging

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)


class BaseModel(object):
    """Parent class of all deep learning models implemented in Saber.

    Attributes:
        config (Config): A Config object which contains a set of harmonized arguments provided in a
            *.ini file and, optionally, from the command line.
        datasets (list): A list containing one or more Dataset objects.
        embeddings (Embeddings): An object containing loaded word embeddings.
        models (nn.Module): A PyTorch model.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        self.config = config  # Hyperparameters and model details
        self.datasets = datasets  # Dataset(s) tied to this instance
        self.embeddings = embeddings  # Pre-trained word embeddings tied to this instance
        self.model = None  # Saber model tied to this instance

        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, model_filepath):
        """Save a PyTorch model to disk at `model_filepath`.

        Saves a PyTorch model to disk, by saving its architecture and weights as a `.bin` file
        at `model_filepath`.

        Args:
            model_filepath (str): Filepath to save the models architecture and weights.

        Returns:
            `model_filepath`, the filepath to which the models architecture and weight were saved.
        """
        torch.save(self.model.state_dict(), model_filepath)

        return model_filepath

    def reset_model(self):
        """Clears and rebuilds the model.

        Clear and rebuilds the model(s) at `self.models`. This is useful, for example, at the end
        of a cross-validation fold.
        """
        torch.cuda.empty_cache()

        self.specify()

    def prepare_for_transfer(self, datasets):
        """Prepares the model at `self.model` for transfer learning by recreating its last layer.
        """
        raise NotImplementedError

    def prune_output_layers(self, indices):
        """Removes output layers with indices not in `output_layer_indices` in `self.model`.

        Args:
            output_layer_indices (int or list): An integer index or list of inter indices into the
                output layers of `self.model` to retain.

        Returns:
            `self.model`, where any output layers with indices not in `output_layer_indices` have
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

        self.model.classifier = nn.ModuleList([self.model.classifier[i] for i in indices])

        return self.model

    # TODO (John): Figure this out!
    def summary(self):
        """Prints a summary representation of the PyTorch model `self.model`.
        """
        pass
