"""Contains the Config class, which is used for parsing and representing all general arguments,
model hyperparameters, and training details.
"""
import argparse
import configparser
import logging
import os

from pkg_resources import resource_filename

from . import constants
from .preprocessor import Preprocessor
from .utils import generic_utils

LOGGER = logging.getLogger(__name__)


class Config(object):
    """A class for managing all hyperparameters and configurations of a model.

    Contains methods for parsing arguments supplied at the command line or in a python ConfigParser
    object. Deals with harmonizing arguments from both of these sources. Each arguments value is
    assigned to an instance attribute.

    Args:
        filepath (str): Path to a *.ini file. If None, default config file is loaded.
        cli (bool): True if command line arguments will be supplied, defaults to False.
    """
    def __init__(self, filepath=None, cli=False):
        # need to parse arguments first in case config filepath provided
        self.cli_args = self._parse_cli_args() if cli else {}
        self.filepath = self._resolve_filepath(filepath, self.cli_args)
        self.config = self._load_config(self.filepath)
        # harmonizing cli and config file arguments
        self.harmonize_args(self.cli_args)

    def save(self, directory):
        """Saves harmonized args sourced from the *.ini file and the command line to `directory`.

        Saves a config.ini file at `directory`, containing the harmonized arguments sourced from the
        original config file at `self.config` and any arguments supplied at the command line.

        Args:
            directory (str): Directory to save the config.ini file.
        """
        # creat filepath to save the config.ini file
        directory = generic_utils.clean_path(directory)
        generic_utils.make_dir(directory)
        filepath = os.path.join(directory, constants.CONFIG_FILENAME)

        with open(filepath, 'w') as config_file:
            for section in self.config.sections():
                # write config file section header
                config_file.write('[{}]\n'.format(section))
                # for each argument in the section, write the argument and its value to the file
                for arg in self.config[section]:
                    value = getattr(self, arg)
                    # need to un-process processed arguments
                    if isinstance(value, list):
                        value = ', '.join(value)
                    elif isinstance(value, dict):
                        value = [str(v) for v in value.values()]
                        value = ', '.join(value)
                    config_file.write('{} = {}\n'.format(arg, value))
                config_file.write('\n')

    def harmonize_args(self, cli_args):
        """Harmonizes args provided via a config file (`self.config`) and command line (`cli_args`).

        Harmonizes the arguments passed to Saber via a config file (`self.config`) with those that
        are optionally provided via the command line (`cli_args`).

        Args:
            cli_args (dict): Dictionary of command line arguments and their values.

        Returns:
            A set of arguments that have been resolved from `self.config` and optionally `cli_args`.
        """
        args = self._parse_config_args(self.config)
        for key, value in cli_args.items():
            # is not False needed to prevent store_true args from overriding corresponding
            # config file args when they are not passed at the command line.
            if value is not None and value is not False:
                args[key] = value
        # post-processing
        args = self._post_process_config_args(args)
        # use parameters dictionary to update instance attributes
        for arg, value in args.items():
            setattr(self, arg, value)

    @classmethod
    def _load_config(self, filepath):
        """Returns a parsed ConfigParser object for config file at 'filepath'.

        Args:
            filepath (str): Path to a *.ini file, which serves as a config file for Saber.

        Returns:
            ConfigParser object, parsed from the *.ini file at `filepath`
        """
        config = configparser.ConfigParser()
        config.read(filepath)

        return config

    @classmethod
    def _resolve_filepath(self, filepath, cli_args):
        """Return appropriate filepath based on how Config class was invoked.

        Resolves the filepath to a *.ini file based on how the Config class was invoked. If
        `cli` was True when the constructor was called, we use the `filepath` provided in
        `cli_args`. If a `filepath` was passed to the constructor and `cli` was False, then we use
        this filepath. Otherwise, we use the default filepath.

        Args:
            filepath (str): Path to a *.ini file, which serves as a config file for Saber.
            cli_args (dict): Dictionary of command line arguments and their values.

        Returns:
            The appropriate filepath to a *.ini file based on how Config class was invoked.
        """
        if cli_args:
            filepath = cli_args['config_filepath']
        elif filepath is None:
            filepath = resource_filename(__name__, constants.CONFIG_FILENAME)

        return filepath

    def _parse_config_args(self, config):
        """Collect arguments from a ConfigParser object parsed from *.ini file at `self.filepath`.

        Args:
            config (ConfigParser): config object parsed from the *.ini file at `self.filepath`.

        Returns:
            A dictionary containing the arguments parsed from a ConfigParser object, `config`.
        """
        args = {}
        try:
            # mode
            args['model_name'] = config['mode']['model_name']
            args['save_model'] = config['mode'].getboolean('save_model')
            # data
            args['dataset_folder'] = config['data']['dataset_folder'].split(',')
            args['output_folder'] = config['data']['output_folder']
            args['pretrained_model'] = config['data']['pretrained_model']
            args['pretrained_embeddings'] = config['data']['pretrained_embeddings']
            # model
            args['word_embed_dim'] = config['model'].getint('word_embed_dim')
            args['char_embed_dim'] = config['model'].getint('char_embed_dim')
            # training
            args['optimizer'] = config['training']['optimizer']
            args['activation'] = config['training']['activation']
            args['learning_rate'] = config['training'].getfloat('learning_rate')
            args['decay'] = config['training'].getfloat('decay')
            args['grad_norm'] = config['training'].getfloat('grad_norm')
            args['dropout_rate'] = config['training']['dropout_rate'].split(',')
            args['batch_size'] = config['training'].getint('batch_size')
            args['k_folds'] = config['training'].getint('k_folds')
            args['epochs'] = config['training'].getint('epochs')
            args['criteria'] = config['training']['criteria']
            # advanced
            args['verbose'] = config['advanced'].getboolean('verbose')
            args['debug'] = config['advanced'].getboolean('debug')
            args['tensorboard'] = config['advanced'].getboolean('tensorboard')
            args['save_all_weights'] = config['advanced'].getboolean('save_all_weights')
            args['replace_rare_tokens'] = config['advanced'].getboolean('replace_rare_tokens')
            args['load_all_embeddings'] = config['advanced'].getboolean('load_all_embeddings')
            args['fine_tune_word_embeddings'] = config['advanced'].getboolean('fine_tune_word_embeddings')
            # TEMP
            args['variational_dropout'] = config['advanced'].getboolean('variational_dropout')
        # ConfigParser throws KeyError when key into object that does not exist
        # catch it and provide hint to the user
        except KeyError as key:
            err_msg = ('KeyError raised for key {}. This may have happened because there is no '
                       '*.ini file at: {}').format(key, self.filepath)
            LOGGER.error('KeyError %s', err_msg)
            print(err_msg)

        LOGGER.debug('Hyperparameters and model details %s', args)

        return args

    @classmethod
    def _post_process_config_args(self, args):
        """Post process parameters retried from python config file.

        Performs series of post processing steps on `args`. E.g., file and directory path arguments
        are normalized, string arguments are cleaned.

        Args:
            args (dict): Contains arguments (keys) and their values (values).

        Returns:
            `args`, where post-processing has been applied to some values.
        """
        # normalize strings
        args['model_name'] = Preprocessor.sterilize(args['model_name'], lower=True)
        args['optimizer'] = Preprocessor.sterilize(args['optimizer'], lower=True)
        args['activation'] = Preprocessor.sterilize(args['activation'], lower=True)
        args['criteria'] = Preprocessor.sterilize(args['criteria'], lower=True)
        # create normalized absolutized versions of paths
        args['dataset_folder'] = [generic_utils.clean_path(ds) for ds in args['dataset_folder']]
        args['output_folder'] = generic_utils.clean_path(args['output_folder'])
        if args['pretrained_model'] and args['pretrained_model'] not in constants.PRETRAINED_MODELS:
            args['pretrained_model'] = generic_utils.clean_path(args['pretrained_model'])
        if args['pretrained_embeddings']:
            args['pretrained_embeddings'] = generic_utils.clean_path(args['pretrained_embeddings'])
        # build dictionary for dropout rates
        args['dropout_rate'] = {
            'input': float(args['dropout_rate'][0]),
            'output': float(args['dropout_rate'][1]),
            'recurrent': float(args['dropout_rate'][2]),
        }

        return args

    @classmethod
    def _parse_cli_args(self):
        """Parse command line arguments passed with call to Saber CLI.

        Returns:
            A dictionary containing all arguments and their values supplied at the command line.
        """
        parser = argparse.ArgumentParser(description='Saber CLI.')

        parser.add_argument('--config_filepath', required=False, type=str,
                            default=resource_filename(__name__, constants.CONFIG_FILENAME),
                            help='Path to the *.ini file containing any arguments')
        parser.add_argument('--activation', required=False, type=str,
                            help=("Activation function to use in the dense layers. Defaults to "
                                  "'relu'."))
        parser.add_argument('--batch_size', required=False, type=int,
                            help=('Integer or None. Number of samples per gradient update.'
                                  'Defaults to 32.'))
        parser.add_argument('--char_embed_dim', required=False, type=str,
                            help='Dimension of dense embeddings to be learned for each character.')
        parser.add_argument('--criteria', required=False, type=str,
                            help=('Matching criteria used to determine true-positives. Choices are '
                                  "'left' for left-boundary matching, 'right' for right-boundary "
                                  "matching and 'exact' for exact-boundary matching."))
        parser.add_argument('--dataset_folder', required=False, nargs='*',
                            help=("Path to the dataset folder. Expects a file 'train.*' to be "
                                  "present. Optionally, 'valid.*' and 'test.*' may be provided. "
                                  "Multiple datasets can be provided, sperated by a space"))
        parser.add_argument('--debug', required=False, action='store_true',
                            help=('If provided, only a small proportion of the dataset, and any '
                                  'provided embeddings, are loaded. Useful for debugging.'))
        parser.add_argument('--decay', required=False, type=float,
                            help=('float >= 0. Learning rate decay over each update. Note that for '
                                  'certain optimizers this value is ignored. Defaults to 0.'))
        parser.add_argument('--dropout_rate', required=False, nargs=3, type=float,
                            metavar=('input', 'output', 'recurrent'),
                            help=('Expects three values, separated by a space, which specify the '
                                  'fraction of units to drop for input, output and recurrent '
                                  'connections respectively. Values must be between 0 and 1.'))
        parser.add_argument('--fine_tune_word_embeddings', required=False, action='store_true',
                            help=('Pass this argument if pre-trained word embedding should be '
                                  'fine-tuned during training.'))
        parser.add_argument('--grad_norm', required=False, type=float,
                            help='Tau threshold value for gradient normalization, defaults to 1.')
        parser.add_argument('--k_folds', required=False, type=int,
                            help=('Number of folds to preform in cross-validation, defaults to 5. '
                                  "Argument is ignored if 'test.*' file present in dataset_folder"))
        parser.add_argument('--learning_rate', required=False, type=float,
                            help=('float >= 0. Learning rate. Note that for certain optimizers '
                                  'this value is ignored'))
        parser.add_argument('--load_all_embeddings', required=False, action='store_true',
                            help=('Pass this argument if all pre-trained embeddings should be '
                                  'loaded, and not just those found in the dataset(s). Has no '
                                  'effect if --pretrained_embeddings argument is empty.'))
        parser.add_argument('--epochs', required=False, type=int,
                            help=('Integer. Number of epochs to train the model. An epoch is an '
                                  'iteration over all data provided.'))
        parser.add_argument('--model_name', required=False, type=str,
                            help="Which model architecture to use. Must be one of ['MT-LSTM-CRF,']")
        parser.add_argument('--optimizer', required=False, type=str,
                            help=("Name of the optimization function to use during training. All "
                                  "optimizers implemented in Keras are supported. Defaults to "
                                  "'nadam'."))
        parser.add_argument('--output_folder', required=False, type=str,
                            help='Path to top-level directory to save all output files.')
        parser.add_argument('--pretrained_model', required=False, type=str,
                            help='Filepath to pre-trained Saber model.')
        parser.add_argument('--replace_rare_tokens', required=False, action='store_true',
                            help=('If True, word types that occur only once in the training '
                                  'dataset will be replaced with a special unknown token.'))
        parser.add_argument('--save_model', required=False, action='store_true',
                            help=('True if the model should be saved when training is complete. '
                                  'Defaults to False.'))
        parser.add_argument('--save_all_weights', required=False, action='store_true',
                            help=('True if the weights from every epoch during training should be '
                                  'saved. If false, weights are only saved for epochs that achieve '
                                  'a new best on validation loss. Defaults to False.'))
        parser.add_argument('--tensorboard', required=False, action='store_true',
                            help=('True if tensorboard logs should be saved during training (note '
                                  'these can be very large). Defaults to False.'))
        parser.add_argument('--word_embed_dim', required=False, type=int,
                            help='Dimension of dense embeddings to be learned for each word.')
        parser.add_argument('--pretrained_embeddings', required=False, type=str,
                            help='Filepath to pre-trained word embeddings.')
        parser.add_argument('--variational_dropout', required=False, action='store_true',
                            help=('Pass this flag if variational dropout should be used. NOTE THAT '
                                  'THIS IS TEMPORARY.'))
        parser.add_argument('--verbose', required=False, action='store_true',
                            help=('True to display more information, such has model details, '
                                  'hyperparameters, and architecture. Defaults to False.'))

        cli_args = parser.parse_args()

        return vars(cli_args)
