"""Any and all unit tests for the generic_utils (saber/utils/generic_utils.py).
"""
import os

import pytest

from ..config import Config
from ..utils import generic_utils
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture(scope='session')
def dummy_dir(tmpdir_factory):
    """Returns the path to a temporary directory.
    """
    dummy_dir = tmpdir_factory.mktemp('dummy_dir')
    return dummy_dir.strpath

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    return dummy_config

############################################ UNIT TESTS ############################################

def test_is_consecutive_empty():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed an
    empty list.
    """
    test = []

    expected = True
    actual = generic_utils.is_consecutive(test)

    assert actual == expected

def test_is_consecutive_simple_sorted_list_no_duplicates():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed a
    simple sorted list with no duplicates.
    """
    test_true = [0, 1, 2, 3, 4, 5]
    test_false = [1, 2, 3, 4, 5, 6]

    expected_true = True
    expected_false = False

    actual_true = generic_utils.is_consecutive(test_true)
    actual_false = generic_utils.is_consecutive(test_false)

    assert actual_true == expected_true
    assert actual_false == expected_false

def test_is_consecutive_simple_unsorted_list_no_duplicates():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed a
    simple unsorted list with no duplicates.
    """
    test_true = [0, 1, 3, 2, 4, 5]
    test_false = [1, 2, 4, 3, 5, 6]

    expected_true = True
    expected_false = False

    actual_true = generic_utils.is_consecutive(test_true)
    actual_false = generic_utils.is_consecutive(test_false)

    assert actual_true == expected_true
    assert actual_false == expected_false

def test_is_consecutive_simple_sorted_list_duplicates():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed a
    simple sorted list with duplicates.
    """
    test = [0, 1, 2, 3, 3, 4, 5]

    expected = False
    actual = generic_utils.is_consecutive(test)

    assert actual == expected

def test_is_consecutive_simple_unsorted_list_duplicates():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed a
    simple unsorted list with duplicates.
    """
    test = [0, 1, 4, 3, 3, 2, 5]

    expected = False
    actual = generic_utils.is_consecutive(test)

    assert actual == expected

def test_is_consecutive_simple_dict_no_duplicates():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed a
    dictionaries values no duplicates.
    """
    test_true = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5}
    test_false = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}

    expected_true = True
    expected_false = False

    actual_true = generic_utils.is_consecutive(test_true.values())
    actual_false = generic_utils.is_consecutive(test_false.values())

    assert actual_true == expected_true
    assert actual_false == expected_false

def test_is_consecutive_simple_dict_duplicates():
    """Asserts that `generic_utils.is_consecutive()` returns the expected value when passed a
    dictionaries values with duplicates.
    """
    test = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 3, 'f': 4, 'g': 5}

    expected = False
    actual = generic_utils.is_consecutive(test)

    assert actual == expected

def test_reverse_dict_empty():
    """Asserts that `generic_utils.reverse_dictionary()` returns the expected value when given an
    empty dictionary.
    """
    test = {}
    expected = {}
    actual = generic_utils.reverse_dict(test)

    assert actual == expected

def test_reverse_mapping_simple():
    """Asserts that `generic_utils.reverse_dictionary()` returns the expected value when given a
    simply dictionary.
    """
    test = {'a': 1, 'b': 2, 'c': 3}

    expected = {1: 'a', 2: 'b', 3: 'c'}
    actual = generic_utils.reverse_dict(test)

    assert actual == expected

def test_make_dir_new(tmpdir):
    """Assert that `generic_utils.make_dir()` creates a directory as expected when it does not
    already exist.
    """
    dummy_dirpath = os.path.join(tmpdir.strpath, 'dummy_dir')
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
