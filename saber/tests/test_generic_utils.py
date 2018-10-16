"""Any and all unit tests for the generic_utils (saber/utils/generic_utils.py).
"""
import os

import pytest

from .. import constants
from ..config import Config
from ..utils import generic_utils
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture(scope='session')
def dummy_dir(tmpdir_factory):
    """Returns the path to a temporary directory.
    """
    dummy_dir = tmpdir_factory.mktemp('dummy_dir')
    return dummy_dir

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    return dummy_config

############################################ UNIT TESTS ############################################

def test_make_dir_new(tmpdir):
    """Assert that `generic_utils.make_dir()` creates a directory as expected when it does not
    already exist.
    """
    dummy_dirpath = os.path.join(tmpdir, 'dummy_dir')
    generic_utils.make_dir(dummy_dirpath)
    assert os.path.isdir(dummy_dirpath)

def test_make_dir_exists(dummy_dir):
    """Assert that `generic_utils.make_dir()` fails silently when trying to create a directory that
    already exists.
    """
    generic_utils.make_dir(dummy_dir)
    assert os.path.isdir(dummy_dir)

def test_clean_path():
    """Asserts that filepath returned by `generic_utils.clean_path()` is as expected.
    """
    test = ' this/is//a/test/     '
    expected = os.path.abspath('this/is/a/test')

    assert generic_utils.clean_path(test) == expected

def test_decompress_model():
    """Asserts that `generic_utils.decompress_model()` decompresses a given directory.
    """
    pass

def test_compress_model():
    """Asserts that `generic_utils.compress_model()` compresses a given directory.
    """
    pass

def test_get_pretrained_model_dir(dummy_config):
    """Asserts that filepath returned by `generic_utils.get_pretrained_model_dir()` is as expected.
    """
    dataset = os.path.basename(dummy_config.dataset_folder[0])
    expected = os.path.join(dummy_config.output_folder, constants.PRETRAINED_MODEL_DIR, dataset)
    assert generic_utils.get_pretrained_model_dir(dummy_config) == expected
