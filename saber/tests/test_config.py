"""Any and all unit tests for the Config class (saber/config.py).
"""
import configparser
from os.path import abspath
from os.path import join
import pytest

from ..config import Config

# constants for dummy config/CL arguments to perform testing on
PATH_TO_DUMMY_CONFIG = abspath('saber/tests/resources/dummy_config.ini')

# Sections of the .ini file
CONFIG_SECTIONS = ['mode', 'data', 'model', 'training', 'advanced']

# Arg values before any processing
DUMMY_ARGS_NO_PROCESSING = {'model_name': 'MT-LSTM-CRF',
                            'train_model': 'True',
                            'save_model': 'False',
                            'dataset_folder': 'saber/tests/resources/dummy_dataset_1',
                            'output_folder': '../output',
                            'pretrained_model_weights': '',
                            'pretrained_embeddings': ('saber/tests/resources/'
                                                      'dummy_word_embeddings/'
                                                      'dummy_word_embeddings.txt'),
                            'word_embed_dim': '200',
                            'char_embed_dim': '30',
                            'optimizer': 'nadam',
                            'activation': 'relu',
                            'learning_rate': '0.0',
                            'grad_norm': '1.0',
                            'decay': '0.0',
                            'dropout_rate': '0.3, 0.3, 0.1',
                            'batch_size': '32',
                            'k_folds': '2',
                            'epochs': '50',
                            'criteria': 'exact',
                            'verbose': 'False',
                            'debug': 'False',
                            'tensorboard': 'False',
                            'replace_rare_tokens': 'False',
                            'fine_tune_word_embeddings': 'False',
                            # TEMP
                            'variational_dropout': 'True',
                           }
# Final arg values when args provided in only config file
DUMMY_ARGS_NO_CLI_ARGS = {'model_name': 'mt-lstm-crf',
                          'train_model': True,
                          'save_model': False,
                          'dataset_folder': [abspath('saber/tests/resources/dummy_dataset_1')],
                          'output_folder': abspath('../output'),
                          'pretrained_model_weights': '',
                          'pretrained_embeddings': abspath(('saber/tests/resources/'
                                                            'dummy_word_embeddings/'
                                                            'dummy_word_embeddings.txt')),
                          'word_embed_dim': 200,
                          'char_embed_dim': 30,
                          'optimizer': 'nadam',
                          'activation': 'relu',
                          'learning_rate': 0.0,
                          'decay': 0.0,
                          'grad_norm': 1.0,
                          'dropout_rate': {'input': 0.3, 'output':0.3, 'recurrent': 0.1},
                          'batch_size': 32,
                          'k_folds': 2,
                          'epochs': 50,
                          'criteria': 'exact',
                          'verbose': False,
                          'debug': False,
                          'tensorboard': False,
                          'replace_rare_tokens': False,
                          'fine_tune_word_embeddings': False,
                          # TEMP
                          'variational_dropout': True,
                         }

# Final arg values when args provided in config file and from CLI
DUMMY_COMMAND_LINE_ARGS = {'optimizer': 'sgd',
                           'grad_norm': 1.0,
                           'learning_rate':0.05,
                           'decay':0.5,
                           'dropout_rate': [0.6, 0.6, 0.2],
                          }
DUMMY_ARGS_WITH_CLI_ARGS = {'model_name': 'mt-lstm-crf',
                            'train_model': True,
                            'save_model': False,
                            'dataset_folder': [abspath('saber/tests/resources/dummy_dataset_1')],
                            'output_folder': abspath('../output'),
                            'pretrained_model_weights': '',
                            'pretrained_embeddings': abspath(('saber/tests/resources/'
                                                              'dummy_word_embeddings/'
                                                              'dummy_word_embeddings.txt')),
                            'word_embed_dim': 200,
                            'char_embed_dim': 30,
                            'optimizer': 'sgd',
                            'activation': 'relu',
                            'learning_rate': 0.05,
                            'decay': 0.5,
                            'grad_norm': 1.0,
                            'dropout_rate': {'input': 0.6, 'output': 0.6, 'recurrent': 0.2},
                            'batch_size': 32,
                            'k_folds': 2,
                            'epochs': 50,
                            'criteria': 'exact',
                            'verbose': False,
                            'debug': False,
                            'tensorboard': False,
                            'fine_tune_word_embeddings': False,
                            'replace_rare_tokens': False,
                            # TEMP
                            'variational_dropout': True,
                           }

@pytest.fixture
def config_no_cli_args():
    """Returns an instance of a Config object after parsing the dummy config file with no command
    line interface (CLI) args."""
    # parse the dummy config
    return Config(filepath=PATH_TO_DUMMY_CONFIG, cli=False)

@pytest.fixture
def config_with_cli_args():
    """Returns an instance of a Config object after parsing the dummy config file with command line
    interface (CLI) args."""
    # parse the dummy config, leave cli false and instead pass the command
    # line args through Config.process_parameters
    config = Config(filepath=PATH_TO_DUMMY_CONFIG, cli=False)
    config._process_args(DUMMY_COMMAND_LINE_ARGS)
    return config

def test_process_args_no_cli_args(config_no_cli_args):
    """Asserts the Config.config object contains the expected values after initializing a Config
    object without CLI args."""
    config = config_no_cli_args.config
    for section in CONFIG_SECTIONS:
        for k, v in config[section].items():
            assert v == DUMMY_ARGS_NO_PROCESSING[k]

def test_process_args_with_cli_args(config_with_cli_args):
    """Asserts the Config.config object contains the expected values after initializing a Config
    object with CLI args."""
    config = config_with_cli_args.config
    for section in CONFIG_SECTIONS:
        for k, v in config[section].items():
            assert v == DUMMY_ARGS_NO_PROCESSING[k]

def test_config_attributes_no_cli_args(config_no_cli_args):
    """Asserts that the class attributes of a Config object are of the expected value/type after
    objects initialization, with NO command line arguments.
    """
    # check that we get the values we expected
    for k, v in DUMMY_ARGS_NO_CLI_ARGS.items():
        assert v == getattr(config_no_cli_args, k)

def test_config_attributes_with_cli_args(config_with_cli_args):
    """Asserts that the class attributes of a Config object are of the expected value/type after
    object initialization, taking into account command line arguments, which take precedence over
    config arguments.
    """
    # check that we get the values we expected, specifically, check that our command line arguments
    # have overwritten our config arguments
    for k, v in DUMMY_ARGS_WITH_CLI_ARGS.items():
        assert v == getattr(config_with_cli_args, k)

def test_save_no_cli_args(config_no_cli_args, tmpdir):
    # save the config to temporary directory created by py.test
    config_no_cli_args.save(tmpdir)
    # load the saved config
    saved_config = load_saved_config(tmpdir)

    # need to 'unprocess' the args to check them against the saved config file
    unprocessed_args = unprocess_args(DUMMY_ARGS_NO_CLI_ARGS)

    # ensure the saved config file matches the original arguments used to create it
    for section in CONFIG_SECTIONS:
        for v, unprocessed_arg in zip(saved_config[section].values(), unprocessed_args):
            assert v == unprocessed_arg

def test_save_with_cli_args(config_with_cli_args, tmpdir):
    # save the config to temporary directory created by py.test
    config_with_cli_args.save(tmpdir)
    # load the saved config
    saved_config = load_saved_config(tmpdir)
    # need to 'unprocess' the args to check them against the saved config file
    unprocessed_args = unprocess_args(DUMMY_ARGS_WITH_CLI_ARGS)

    # ensure the saved config file matches the original arguments used to create it
    for section in CONFIG_SECTIONS:
        for v, unprocessed_arg in zip(saved_config[section].values(), unprocessed_args):
            assert v == unprocessed_arg

# helper functions
def load_saved_config(filepath):
    """Load a saved ConfigParser object at 'filepath/config.ini'.

    Args:
        filepath (str): filepath to the saved config file 'config.ini'

    Returns:
        parsed ConfigParser object at 'filepath/config.ini'.
    """
    saved_config_filepath = join(filepath, 'config.ini')
    saved_config = configparser.ConfigParser()
    saved_config.read(saved_config_filepath)

    return saved_config

def unprocess_args(args):
    """Unprocesses processed config args.

    Given a dictionary of arguments ('arg'), returns a list containing all argument values where
    all values have been converted to strings.

    Returns:
        a list, containing all elements of `args.values()` converted to a str representation.
    """
    unprocessed_args = []
    for arg in args:
        if isinstance(arg, list):
            unprocessed_arg = ', '.join(arg)
        elif isinstance(arg, dict):
            values = [str(v) for v in arg.values()]
            unprocessed_arg = ', '.join(values)
        else:
            unprocessed_arg = str(arg)
    return unprocessed_args
