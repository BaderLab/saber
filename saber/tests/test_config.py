"""Contains any and all unit tests for the Config class (saber/config.py).
"""
import configparser
import os

import pytest

from ..config import Config
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def config_no_cli_args():
    """Returns an instance of a Config object after parsing the dummy config file with no command
    line interface (CLI) args."""
    # the dataset and embeddings are used for test purposes so they must point to the
    # correct resources, this can be ensured by passing their paths here
    cli_arguments = {'dataset_folder': [PATH_TO_DUMMY_DATASET_1],
                     'pretrained_embeddings': PATH_TO_DUMMY_EMBEDDINGS}
    # parse the dummy config
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def config_with_cli_args():
    """Returns an instance of a Config object after parsing the dummy config file with command line
    interface (CLI) args."""
    # parse the dummy config, leave cli false and instead pass command line args manually
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    # this is a bit of a hack, but need to simulate providing commands at the command line
    dummy_config.cli_args = DUMMY_COMMAND_LINE_ARGS
    dummy_config._process_args(DUMMY_COMMAND_LINE_ARGS)

    return dummy_config

############################################ UNIT TESTS ############################################

def test_process_args_no_cli_args(config_no_cli_args):
    """Asserts the Config.config object contains the expected attributes after initializing a Config
    object without CLI args."""
    # check filepath attribute
    assert config_no_cli_args.filepath == PATH_TO_DUMMY_CONFIG
    # check that the config file contains the same values as DUMMY_ARGS_NO_PROCESSING
    config = config_no_cli_args.config
    for section in CONFIG_SECTIONS:
        for arg, value in config[section].items():
            assert value == DUMMY_ARGS_NO_PROCESSING[arg]
    # check cli_args attribute
    assert config_no_cli_args.cli_args == {}

def test_process_args_with_cli_args(config_with_cli_args):
    """Asserts the Config.config object contains the expected attributes after initializing a Config
    object with CLI args."""
    # check filepath attribute
    assert config_with_cli_args.filepath == os.path.join(os.path.dirname( \
        os.path.os.path.abspath(__file__)), PATH_TO_DUMMY_CONFIG)
    config = config_with_cli_args.config
    # check that the config file contains the same values as DUMMY_ARGS_NO_PROCESSING
    for section in CONFIG_SECTIONS:
        for arg, value in config[section].items():
            assert value == DUMMY_ARGS_NO_PROCESSING[arg]
    # check cli_args attribute
    assert config_with_cli_args.cli_args == DUMMY_COMMAND_LINE_ARGS

def test_config_attributes_no_cli_args(config_no_cli_args):
    """Asserts that the class attributes of a Config object are of the expected value/type after
    objects initialization, with NO command line arguments.
    """
    # check that we get the values we expected
    for arg, value in DUMMY_ARGS_NO_CLI_ARGS.items():
        assert value == getattr(config_no_cli_args, arg)

def test_config_attributes_with_cli_args(config_with_cli_args):
    """Asserts that the class attributes of a Config object are of the expected value/type after
    object initialization, taking into account command line arguments, which take precedence over
    config arguments.
    """
    # check that we get the values we expected, specifically, check that our command line arguments
    # have overwritten our config arguments
    for arg, value in DUMMY_ARGS_WITH_CLI_ARGS.items():
        assert value == getattr(config_with_cli_args, arg)

def test_save_no_cli_args(config_no_cli_args, tmpdir):
    """Asserts that a saved config file contains the correct arguments and values."""
    # save the config to temporary directory created by py.test
    config_no_cli_args.save(tmpdir)
    # load the saved config
    saved_config = load_saved_config(tmpdir)
    # need to 'unprocess' the args to check them against the saved config file
    unprocessed_args = unprocess_args(DUMMY_ARGS_NO_CLI_ARGS)
    # ensure the saved config file matches the original arguments used to create it
    for section in CONFIG_SECTIONS:
        for arg, value in saved_config[section].items():
            assert value == unprocessed_args[arg]

def test_save_with_cli_args(config_with_cli_args, tmpdir):
    """Asserts that a saved config file contains the correct arguments and values, taking into
    account command line arguments, which take precedence over config arguments.
    """
    # save the config to temporary directory created by py.test
    config_with_cli_args.save(tmpdir)
    # load the saved config
    saved_config = load_saved_config(tmpdir)
    # need to 'unprocess' the args to check them against the saved config file
    unprocessed_args = unprocess_args(DUMMY_ARGS_WITH_CLI_ARGS)
    # ensure the saved config file matches the original arguments used to create it
    for section in CONFIG_SECTIONS:
        for arg, value in saved_config[section].items():
            assert value == unprocessed_args[arg]

######################################### HELPER FUNCTIONS #########################################

def load_saved_config(filepath):
    """Load a saved ConfigParser object at 'filepath/config.ini'.

    Args:
        filepath (str): filepath to the saved config file 'config.ini'

    Returns:
        parsed ConfigParser object at 'filepath/config.ini'.
    """
    saved_config_filepath = os.path.join(filepath, 'config.ini')
    saved_config = configparser.ConfigParser()
    saved_config.read(saved_config_filepath)

    return saved_config

def unprocess_args(args):
    """Unprocesses processed config args.

    Given a dictionary of arguments ('arg'), returns a dictionary where all values have been
    converted to string representation.

    Returns:
        args, where all values have been replaced by a str representation.
    """
    unprocessed_args = {}
    for arg, value in args.items():
        if isinstance(value, list):
            unprocessed_arg = ', '.join(value)
        elif isinstance(value, dict):
            dict_values = [str(v) for v in value.values()]
            unprocessed_arg = ', '.join(dict_values)
        else:
            unprocessed_arg = str(value)

        unprocessed_args[arg] = unprocessed_arg

    return unprocessed_args
