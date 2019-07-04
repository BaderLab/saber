"""Contains any and all unit tests for the config.Config class (saber/config.py).
"""
import os

import pytest
from pkg_resources import resource_filename

from .. import config
from ..constants import CONFIG_FILENAME
from .resources.constants import (CONFIG_SECTIONS, DUMMY_ARGS_NO_CLI_ARGS,
                                  DUMMY_ARGS_NO_PROCESSING,
                                  DUMMY_ARGS_WITH_CLI_ARGS,
                                  DUMMY_COMMAND_LINE_ARGS,
                                  PATH_TO_DUMMY_CONFIG)
from .resources.helpers import load_saved_config, unprocess_args


class TestConfig(object):
    """Collects all unit tests for `saber.utils.bert_utils`.
    """
    def test_process_args_no_cli_args(self, dummy_config):
        """Asserts the config.Config.config object contains the expected attributes after initializing
        a config.Config object without CLI args."""
        # check filepath attribute
        assert dummy_config.filepath == PATH_TO_DUMMY_CONFIG
        # check that the config file contains the same values as DUMMY_ARGS_NO_PROCESSING
        config_ = dummy_config.config
        for section in CONFIG_SECTIONS:
            for arg, value in config_[section].items():
                assert value == DUMMY_ARGS_NO_PROCESSING[arg]
        # check cli_args attribute
        assert dummy_config.cli_args == {}

    def test_process_args_with_cli_args(self, dummy_config_cli_args):
        """Asserts the config.Config.config object contains the expected attributes after initializing a
        config.Config object with CLI args."""
        # check filepath attribute
        assert dummy_config_cli_args.filepath == os.path.join(os.path.dirname(
            os.path.os.path.abspath(__file__)), PATH_TO_DUMMY_CONFIG)
        config_ = dummy_config_cli_args.config
        # check that the config file contains the same values as DUMMY_ARGS_NO_PROCESSING
        for section in CONFIG_SECTIONS:
            for arg, value in config_[section].items():
                assert value == DUMMY_ARGS_NO_PROCESSING[arg]
        # check cli_args attribute
        assert dummy_config_cli_args.cli_args == DUMMY_COMMAND_LINE_ARGS

    def test_config_attributes_no_cli_args(self, dummy_config):
        """Asserts that the class attributes of a config.Config object are of the expected value/type
        after objects initialization, with NO command line arguments.
        """
        # check that we get the values we expected
        for arg, value in DUMMY_ARGS_NO_CLI_ARGS.items():
            assert value == getattr(dummy_config, arg)

    def test_config_attributes_with_cli_args(self, dummy_config_cli_args):
        """Asserts that the class attributes of a config.Config object are of the expected value/type
        after object initialization, taking into account command line arguments, which take
        precedence over config arguments.
        """
        # check that we get the values we expected, specifically, check that our command line
        # arguments have overwritten our config arguments
        for arg, value in DUMMY_ARGS_WITH_CLI_ARGS.items():
            assert value == getattr(dummy_config_cli_args, arg)

    def test_resolve_filepath(self, dummy_config):
        """Asserts that `Config._resolve_filepath()` returns the expected values.
        """
        # tests for when neither filepath nor cli_args arguments are provided
        filepath_none_cli_args_none_expected = \
            resource_filename(config.__name__, CONFIG_FILENAME)
        filepath_none_cli_args_none_actual = \
            dummy_config._resolve_filepath(filepath=None, cli_args={})
        # tests for when cli_args argument is provided
        filepath_none_cli_args_expected = 'arbitrary/filepath/to/config.ini'
        dummy_cli_args = {'config_filepath': filepath_none_cli_args_expected}
        filepath_none_cli_args_actual = dummy_config._resolve_filepath(filepath=None,
                                                                       cli_args=dummy_cli_args)
        # tests for when filepath argument is provided
        filepath_cli_args_none_expected = filepath_none_cli_args_expected
        filepath_cli_args_none_actual = \
            dummy_config._resolve_filepath(filepath=filepath_cli_args_none_expected, cli_args={})
        # tests for when both filepath and cli_args arguments are provided
        filepath_cli_args_expected = filepath_none_cli_args_expected
        filepath_cli_args_actual = \
            dummy_config._resolve_filepath(filepath=filepath_cli_args_expected,
                                           cli_args=dummy_cli_args)

        assert filepath_none_cli_args_none_expected == filepath_none_cli_args_none_actual
        assert filepath_none_cli_args_expected == filepath_none_cli_args_actual
        assert filepath_cli_args_none_expected == filepath_cli_args_none_actual
        assert filepath_cli_args_expected == filepath_cli_args_actual

    def test_key_error(self, tmpdir):
        """Assert that a KeyError is raised when Config object is initialized with a value for
        `filepath` that does does contain a valid *.ini file.
        """
        with pytest.raises(KeyError):
            _ = config.Config(tmpdir.strpath)

    def test_save_no_cli_args(self, dummy_config, tmpdir):
        """Asserts that a saved config file contains the correct arguments and values."""
        # save the config to temporary directory created by py.test
        dummy_config.save(tmpdir.strpath)
        # load the saved config
        saved_config = load_saved_config(tmpdir.strpath)
        # need to 'unprocess' the args to check them against the saved config file
        unprocessed_args = unprocess_args(DUMMY_ARGS_NO_CLI_ARGS)
        # ensure the saved config file matches the original arguments used to create it
        for section in CONFIG_SECTIONS:
            for arg, value in saved_config[section].items():
                assert value == unprocessed_args[arg]

    def test_save_with_cli_args(self, dummy_config_cli_args, tmpdir):
        """Asserts that a saved config file contains the correct arguments and values, taking into
        account command line arguments, which take precedence over config arguments.
        """
        # save the config to temporary directory created by py.test
        dummy_config_cli_args.save(tmpdir.strpath)
        # load the saved config
        saved_config = load_saved_config(tmpdir.strpath)
        # need to 'unprocess' the args to check them against the saved config file
        unprocessed_args = unprocess_args(DUMMY_ARGS_WITH_CLI_ARGS)
        # ensure the saved config file matches the original arguments used to create it
        for section in CONFIG_SECTIONS:
            for arg, value in saved_config[section].items():
                assert value == unprocessed_args[arg]
