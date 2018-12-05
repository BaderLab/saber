"""Any and all helper functions for Sabers unit tests.
"""
import configparser
import os

from ... import constants


def assert_type_to_idx_as_expected(actual, expected):
    """Asserts that a `type_to_idx` mapping is as expected. This involves checking that it contains
    the expected, keys, the expected values, and that the values are a consecutive mapping of
    integers beginning at 0.
    """
    # check keys
    assert all(word in actual['word'] for word in expected['word'])
    assert all(char in actual['char'] for char in expected['char'])
    # check that values are consectutive mapping of integers between 0 and length of the dictionary
    assert all(id in range(0, len(actual['word'])) for id in actual['word'].values())
    assert all(id in range(0, len(actual['char'])) for id in actual['char'].values())
    # check initial mapping items
    assert all(word in actual['word'] for word in constants.INITIAL_MAPPING['word'])
    assert all(word in actual['char'] for word in constants.INITIAL_MAPPING['word'])

def load_saved_config(filepath):
    """Load a saved config.ConfigParser object at 'filepath/config.ini'.

    Args:
        filepath (str): filepath to the saved config file 'config.ini'

    Returns:
        parsed config.ConfigParser object at 'filepath/config.ini'.
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
