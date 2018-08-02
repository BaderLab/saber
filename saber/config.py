"""Contains the Config class, which is used for parsing and representing all general arguments,
model hyperparameters, and training details.
"""
import os
import argparse
import configparser

from .preprocessor import Preprocessor

# TODO: Some arguments still need help strings written

class Config(object):
    """A class for managing all hyperparameters and configurations of a model.

    Conatains methods for parsing arguments supplied at the command line or in a python ConfigParser
    object. Deals with harmonizing arguments from both of these sources. Each arguments value is
    assigned to an instance variable.

    Args:
        filepath (str): path to a .ini file, defaults to ./config.ini
        cli (bool): True if command line arguments will be supplied, defaults to False.
    """
    def __init__(self, filepath='./config.ini', cli=False):
        # filepath to config file
        self.filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
        # a parsed configparser object
        self.config = None

        # parse args provided in configuration file
        self.config = self._parse_config_args(self.filepath)
        # parse cli arguments (if they exist)
        self.cli_args = self._parse_cli_args() if cli else {}
        # harmonize cli and config arguments and apply post processing
        self.args = self._process_args(self.cli_args)

    def _parse_config_args(self, filepath):
        """Returns a parsed configparser object for config file at 'filepath'.

        Args:
            filepath: filepath to .ini config file

        Returns:
            ConfigParser object, parsed from .ini file at 'filepath'
        """
        config = configparser.ConfigParser()
        config.read(filepath)

        return config

    def _process_args(self, cli_args):
        """Collect arguments from ini file if specificed.

        Loads parameters from configparser at 'self.config'. Any identically
        named arguments provided at the command line (provided as a dictionary
        in 'cli_args') will overwrite those found in the configparser object.
        Uses this final set of arguments to initialize and assign identically
        named instance variables.

        Args:
            cli_args (dict): contains any arguments specified at the
                command line (keys) and their values (values).

        Returns:
            a dictionary containing the final harmonized and post processed arguments.
        """
        args = {}

        try:
            # parse config
            # mode
            args['model_name'] = self.config['mode']['model_name']
            args['train_model'] = self.config['mode'].getboolean('train_model')
            args['save_model'] = self.config['mode'].getboolean('save_model')
            args['load_pretrained_model'] = self.config['mode'].getboolean('load_pretrained_model')

            # data
            args['dataset_folder'] = self.config['data']['dataset_folder'].split(',')
            args['output_folder'] = self.config['data']['output_folder']
            args['pretrained_model_weights'] = self.config['data']['pretrained_model_weights']
            args['pretrained_embeddings'] = self.config['data']['pretrained_embeddings']

            # model
            args['word_embed_dim'] = self.config['model'].getint('word_embed_dim')
            args['char_embed_dim'] = self.config['model'].getint('char_embed_dim')

            # training
            args['optimizer'] = self.config['training']['optimizer']
            args['activation'] = self.config['training']['activation']
            args['learning_rate'] = self.config['training'].getfloat('learning_rate')
            args['decay'] = self.config['training'].getfloat('decay')
            args['grad_norm'] = self.config['training'].getfloat('grad_norm')
            args['dropout_rate'] = self.config['training']['dropout_rate'].split(',')
            args['batch_size'] = self.config['training'].getint('batch_size')
            args['k_folds'] = self.config['training'].getint('k_folds')
            args['epochs'] = self.config['training'].getint('epochs')
            args['criteria'] = self.config['training']['criteria']

            # advanced
            args['verbose'] = self.config['advanced'].getboolean('verbose')
            args['debug'] = self.config['advanced'].getboolean('debug')
            args['tensorboard'] = self.config['advanced'].getboolean('tensorboard')
            args['replace_rare_tokens'] = self.config['advanced'].getboolean('replace_rare_tokens')
            args['fine_tune_word_embeddings'] = \
                self.config['advanced'].getboolean('fine_tune_word_embeddings')
            # TEMP
            args['variational_dropout'] = self.config['advanced'].getboolean('variational_dropout')

        # Configparser throws a KeyError when you try to key into a configparser object that does
        # not exist, catch it here, provide hint to the user
        except KeyError as key:
            print('[ERROR] KeyError raised for key {}.'.format(key))
            print(('[WARN] This may have happened because there is no .ini file at: '
                   '{}'.format(self.filepath)))
        else:
            # overwrite any parameters in the config if specfied at CL
            for key, value in cli_args.items():
                if value is not None:
                    args[key] = value

            # post-processing
            args = self._post_process_args(args)

            # use parameters dictionary to update instance attributes
            for arg, value in args.items():
                setattr(self, arg, value)

        return args

    def _post_process_args(self, args):
        """Post process parameters retrived from python config file.

        Performs series of post processing steps on 'parameters'. E.g., file and
        directory path arguments are normalized, str arguments are cleaned.

        Args:
            args (dict): contains arguments (keys) and their values (values).

        Returns:
            args, where post-processing has been applied to some values.
        """
        # normalize strings
        args['model_name'] = Preprocessor.sterilize(args['model_name'], lower=True)
        args['optimizer'] = Preprocessor.sterilize(args['optimizer'], lower=True)
        args['activation'] = Preprocessor.sterilize(args['activation'], lower=True)
        args['criteria'] = Preprocessor.sterilize(args['criteria'], lower=True)

        # create normalized absolutized versions of paths
        args['dataset_folder'] = [os.path.abspath(Preprocessor.sterilize(ds))
                                  for ds in args['dataset_folder']]
        args['output_folder'] = os.path.abspath(args['output_folder'])

        # Normalize empty path arguments to None
        args['pretrained_model_weights'] = None if args['pretrained_model_weights'] == '' \
            else os.path.abspath(args['pretrained_model_weights'])

        args['pretrained_embeddings'] = None if args['pretrained_embeddings'] == '' \
            else os.path.abspath(args['pretrained_embeddings'])

        # build dictionary for dropout rates
        args['dropout_rate'] = {
            'input': float(args['dropout_rate'][0]),
            'output': float(args['dropout_rate'][1]),
            'recurrent': float(args['dropout_rate'][2]),
        }

        return args

    def _parse_cli_args(self):
        """Parse command line arguments passed with call to Saber.

        Returns:
            a dictionary containing all arguments and their values supplied at the command line.
        """
        parser = argparse.ArgumentParser(description='Saber CLI.')

        parser.add_argument('--filepath', required=False, type=str,
                            help=('Path to the .ini file containing any arguments. Defaults to '
                                  './config.ini.'))
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
                                  "present. Optionally, 'valid.*' and 'train.*' may be provided."
                                  "Multiple datasets can be provided, sperated by a space"))
        parser.add_argument('--debug', required=False, action='store_true',
                            help=('If provided, only a small proportion of the dataset, and any '
                                  'provided embeddings, are loaded. Useful for debugging.'))
        parser.add_argument('--decay', required=False, type=float,
                            help=('float >= 0. Learning rate decay over each update. Note that for '
                                  'certain optimizers this value is ignored. Defaults to 0.'))
        parser.add_argument('--dropout_rate', required=False, nargs=3, type=float,
                            metavar=('input', 'output', 'recurrent'),
                            help=('Expectes three values, seperated by a space, which specify the '
                                  'fraction of units to drop for input, output and recurrent '
                                  'connections respectively. Values must be between 0 and 1.'))
        parser.add_argument('--fine_tune_word_embeddings', required=False, action='store_true',
                            help=('Pass this argument if pre-trained word embedding should be '
                                  'fine-tuned during training.'))
        parser.add_argument('--grad_norm', required=False, type=float,
                            help='Tau threshold value for gradient normalization, defaults to 1.')
        parser.add_argument('--k_folds', required=False, type=int,
                            help=('Number of folds to preform in cross-validation, defaults to 5.'
                                  "Argument is ignored if 'test.*' file present in dataset_folder"))
        parser.add_argument('--learning_rate', required=False, type=float,
                            help=('float >= 0. Learning rate. Note that for certain optimizers '
                                  'this value is ignored'))
        parser.add_argument('--load_pretrained_model', required=False, action='store_true',
                            help='TODO')
        parser.add_argument('--epochs', required=False, type=int,
                            help=('Integer. Number of epochs to train the model. An epoch is an '
                                  'iteration over all data provided.'))
        parser.add_argument('--model_name', required=False, type=str,
                            help=('Which model architecture to use. Currently, only MT-LSTM-CRF is '
                                  'provided.'))
        parser.add_argument('--optimizer', required=False, type=str,
                            help=("Name of the optimization function to use during training. All "
                                  'optimizers implemented in Keras are supported. Defaults to '
                                  "'nadam'."))
        parser.add_argument('--output_folder', required=False, type=str,
                            help='Path to the top-level of the directory to save all output files.')
        parser.add_argument('--pretrained_model_weights', required=False, type=str, help='TODO')
        parser.add_argument('--replace_rare_tokens', required=False, action='store_true',
                            help=('If True, word types that occur less than 1 time in the training '
                                  'dataset will be replaced with a special unknown token.'))
        parser.add_argument('--save_model', required=False, action='store_true', help='TODO')
        parser.add_argument('--tensorboard', required=False, action='store_true', help='TODO')
        parser.add_argument('--word_embed_dim', required=False, type=int,
                            help='Dimension of dense embeddings to be learned for each word.')
        parser.add_argument('--pretrained_embeddings', required=False, type=str,
                            help='Filepath to pretrained word embeddings.')
        parser.add_argument('--train_model', required=False, action='store_true', help='TODO')
        parser.add_argument('--variational_dropout', required=False, action='store_true',
                            help=('Pass this flag if variational dropout should be used. NOTE THAT'
                                  'THIS IS TEMPORARY'))
        parser.add_argument('--verbose', required=False, action='store_true', help='TODO')

        # parse cli args, return dictionary representation of these args
        cli_args = parser.parse_args()

        return vars(cli_args)
