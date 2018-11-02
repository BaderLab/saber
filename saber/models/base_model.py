"""Contains the BaseModel class, the parent class to all Keras models in Saber.
"""
import json
import logging

from keras import optimizers
from keras.models import model_from_json

LOGGER = logging.getLogger(__name__)

class BaseKerasModel(object):
    """Parent class of all Keras model classes implemented by Saber.
    """
    def __init__(self, config, datasets, embeddings=None, **kwargs):
        self.config = config # hyperparameters and model details
        self.datasets = datasets # dataset(s) tied to this instance
        self.embeddings = embeddings # pre-trained word embeddings tied to this instance
        self.models = [] # Keras model(s) tied to this instance

        for key, value in kwargs.items():
            setattr(self, key, value)

    def save(self, weights_filepath, model_filepath, model=0):
        """Save a model to disk.

        Saves a keras model to disk, by saving its architecture as a json file at `model_filepath`
        and its weights as a hdf5 file at `model_filepath`.

        Args:
            weights_filepath (str): filepath to the models wieghts (.hdf5 file).
            model_filepath (str): filepath to the models architecture (.json file).
            model (int): which model from `self.models` to save.
        """
        with open(model_filepath, 'w') as f:
            model_json = self.models[model].to_json()
            json.dump(json.loads(model_json), f, sort_keys=True, indent=4)
            self.models[model].save_weights(weights_filepath)

    def load(self, weights_filepath, model_filepath):
        """Load a model from disk.

        Loads a keras model from disk by loading its architecture from a json file at `model_filepath`
        and its weights from a hdf5 file at `model_filepath`.

        Args:
            weights_filepath (str): filepath to the models weights (.hdf5 file).
            model_filepath (str): filepath to the models architecture (.json file).
        """
        with open(model_filepath) as f:
            model = model_from_json(f.read())
            model.load_weights(weights_filepath)
            self.models.append(model)

    def prepare_data_for_training(self):
        """Returns a list containing the training data for each dataset at `self.datasets`.

        For each dataset at `self.datasets`, collects the data to be used for training.
        Each dataset is represented by a dictionary, where the keys 'x_<partition>' and
        'y_<partition>' contain the inputs and targets for each partition 'train', 'valid', and
        'test'.
        """
        training_data = []
        for ds in self.datasets:
            # collect train data, must be provided
            x_train = [ds.idx_seq['train']['word'], ds.idx_seq['train']['char']]
            y_train = ds.idx_seq['train']['tag']
            # collect valid and test data, may not be provided
            x_valid, y_valid, x_test, y_test = None, None, None, None
            if ds.idx_seq['valid'] is not None:
                x_valid = [ds.idx_seq['valid']['word'], ds.idx_seq['valid']['char']]
                y_valid = ds.idx_seq['valid']['tag']
            if ds.idx_seq['test'] is not None:
                x_test = [ds.idx_seq['test']['word'], ds.idx_seq['test']['char']]
                y_test = ds.idx_seq['test']['tag']

            training_data.append({'x_train': x_train, 'y_train': y_train, 'x_valid': x_valid,
                                  'y_valid': y_valid, 'x_test': x_test, 'y_test': y_test})

        return training_data

    def _compile(self, model, loss_function, optimizer, lr=0.01, decay=0.0, clipnorm=0.0):
        """Compiles a model specified with Keras.

        See https://keras.io/optimizers/ for more info on each optimizer.

        Args:
            model: Keras model object to compile
            loss_function: Keras loss_function object to compile model with
            optimizer (str): the optimizer to use during training
            lr (float): learning rate to use during training
            decay (float): per epoch decay rate
            clipnorm (float): gradient normalization threshold
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
