"""Any and all unit tests for the data_utils (saber/utils/data_utils.py).
"""
import numpy as np

import pytest

from ..config import Config
from ..dataset import Dataset
from ..utils import data_utils
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    return dummy_config

@pytest.fixture
def dummy_dataset_1():
    """Returns a single dummy Dataset instance after calling Dataset.load().
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False)
    dataset.load()

    return dataset

@pytest.fixture
def dummy_dataset_2():
    """Returns a single dummy Dataset instance after calling `Dataset.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET_2, replace_rare_tokens=False)
    dataset.load()

    return dataset

@pytest.fixture
def dummy_compound_dataset(dummy_config):
    """
    """
    dummy_config.dataset_folder = [PATH_TO_DUMMY_DATASET_1, PATH_TO_DUMMY_DATASET_2]
    dummy_config.replace_rare_tokens = False
    dataset = data_utils.load_compound_dataset(dummy_config)

    return dataset

@pytest.fixture(scope='session')
def dummy_dataset_paths_all(tmpdir_factory):
    """Creates and returns the path to a temporary dataset folder, and train, valid, test files.
    """
    # create a dummy dataset folder
    dummy_dir = tmpdir_factory.mktemp('dummy_dataset')
    # create train, valid and train partitions in this folder
    train_file = dummy_dir.join('train.tsv')
    train_file.write('arbitrary') # need to write content or else the file wont exist
    valid_file = dummy_dir.join('valid.tsv')
    valid_file.write('arbitrary')
    test_file = dummy_dir.join('test.tsv')
    test_file.write('arbitrary')

    return dummy_dir.strpath, train_file.strpath, valid_file.strpath, test_file.strpath

@pytest.fixture(scope='session')
def dummy_dataset_paths_no_valid(tmpdir_factory):
    """Creates and returns the path to a temporary dataset folder, and train, and test files.
    """
    # create a dummy dataset folder
    dummy_dir = tmpdir_factory.mktemp('dummy_dataset')
    # create train, valid and train partitions in this folder
    train_file = dummy_dir.join('train.tsv')
    train_file.write('arbitrary') # need to write content or else the file wont exist
    test_file = dummy_dir.join('test.tsv')
    test_file.write('arbitrary')

    return dummy_dir.strpath, train_file.strpath, test_file.strpath

############################################ UNIT TESTS ############################################

def test_get_filepaths_value_error(tmpdir):
    """Asserts that a ValueError is raised when `data_utils.get_filepaths(tmpdir)` is called and
    no file '<tmpdir>/train.*' exists.
    """
    with pytest.raises(ValueError):
        data_utils.get_filepaths(tmpdir.strpath)

def test_get_filepaths_all(dummy_dataset_paths_all):
    """Asserts that `data_utils.get_filepaths()` returns the expected filepaths when all partitions
    (train/test/valid) are provided.
    """
    dummy_dataset_directory, train_filepath, valid_filepath, test_filepath = dummy_dataset_paths_all
    expected = {'train': train_filepath,
                'valid': valid_filepath,
                'test': test_filepath
               }
    actual = data_utils.get_filepaths(dummy_dataset_directory)

    assert actual == expected

def test_get_filepaths_no_valid(dummy_dataset_paths_no_valid):
    """Asserts that `data_utils.get_filepaths()` returns the expected filepaths when train and
    test partitions are provided.
    """
    dummy_dataset_directory, train_filepath, test_filepath = dummy_dataset_paths_no_valid
    expected = {'train': train_filepath,
                'valid': None,
                'test': test_filepath
               }
    actual = data_utils.get_filepaths(dummy_dataset_directory)

    assert actual == expected

def test_load_single_dataset(dummy_config, dummy_dataset_1):
    """Asserts that `data_utils.load_single_dataset()` returns the expected value.
    """
    actual = data_utils.load_single_dataset(dummy_config)
    expected = [dummy_dataset_1]

    # essentially redundant, but if we dont return a [Dataset] object then the error message from
    # the final test could be cryptic
    assert isinstance(actual, list)
    assert len(actual) == 1
    assert isinstance(actual[0], Dataset)
    # the test we actually care about, least roundabout way of asking if the two Dataset objects
    # are identical
    assert dir(actual[0].__dict__) == dir(expected[0].__dict__)

def test_load_compound_dataset_unchanged_attributes(dummy_dataset_1,
                                                    dummy_dataset_2,
                                                    dummy_compound_dataset):
    """Asserts that attributes of `Dataset` objects which are expected to remain unchanged
    are unchanged after call to `data_utils.load_compound_dataset()`.
    """
    actual = dummy_compound_dataset
    expected = [dummy_dataset_1, dummy_dataset_2]

    # essentially redundant, but if we dont return a [Dataset, Dataset] object then the error
    # messages from the downstream tests could be cryptic
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert all([isinstance(ds, Dataset) for ds in actual])

    # attributes that are unchanged in case of compound dataset
    assert actual[0].directory == expected[0].directory
    assert actual[0].replace_rare_tokens == expected[0].replace_rare_tokens
    assert actual[0].type_seq == expected[0].type_seq
    assert actual[0].type_to_idx['tag'] == expected[0].type_to_idx['tag']
    assert actual[0].idx_to_tag == expected[0].idx_to_tag

    assert actual[-1].directory == expected[-1].directory
    assert actual[-1].replace_rare_tokens == expected[-1].replace_rare_tokens
    assert actual[-1].type_seq == expected[-1].type_seq
    assert actual[-1].type_to_idx['tag'] == expected[-1].type_to_idx['tag']
    assert actual[-1].idx_to_tag == expected[-1].idx_to_tag

def test_load_compound_dataset_changed_attributes(dummy_dataset_1,
                                                  dummy_dataset_2,
                                                  dummy_compound_dataset):
    """Asserts that attributes of `Dataset` objects which are expected to be changed are changed
    after call to `data_utils.load_compound_dataset()`.
    """
    actual = dummy_compound_dataset
    expected = [dummy_dataset_1, dummy_dataset_2]

    # essentially redundant, but if we don't return a [Dataset, Dataset] object then the error
    # messages from the downstream tests could be cryptic
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert all([isinstance(ds, Dataset) for ds in actual])

    # attributes that are changed in case of compound dataset
    assert actual[0].type_to_idx['word'] == actual[-1].type_to_idx['word']
    assert actual[0].type_to_idx['char'] == actual[-1].type_to_idx['char']

    # TODO: Need to assert that all types in idx_seq map to the same integers
    # across the compound datasets

def test_setup_dataset_for_transfer(dummy_dataset_1, dummy_dataset_2):
    """Asserts that the `type_to_idx` attribute of a "source" dataset and a "target" dataset are
    as expected after call to `data_utils.setup_dataset_for_transfer()`.
    """
    source_type_to_idx = dummy_dataset_1.type_to_idx
    data_utils.setup_dataset_for_transfer(dummy_dataset_2, source_type_to_idx)

    assert all(dummy_dataset_2.type_to_idx[type_] == source_type_to_idx[type_] for type_ in ['word', 'char'])
