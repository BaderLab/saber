"""Any and all unit tests for the data_utils (saber/utils/data_utils.py).
"""
import numpy as np

import pytest

from ..utils import data_utils

######################################### PYTEST FIXTURES #########################################

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

    return dummy_dir, train_file, valid_file, test_file

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

    return dummy_dir, train_file, test_file

############################################ UNIT TESTS ############################################

def test_get_filepaths_value_error(tmpdir):
    """Asserts that a ValueError is raised when `data_utils.get_filepaths(tmpdir)` is called and
    no file '<tmpdir>/train.*' exists.
    """
    with pytest.raises(ValueError):
        data_utils.get_filepaths(tmpdir)

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

def test_one_hot_encode():
    """Asserts that the one-hot encoding returned by `data_utils.one_hot_encode()` is as expected.
    """
    # empty list test
    empty_test = []
    # simple list test and its expected value
    simple_test = [[0, 1, 3], [2, 1, 3]]
    simple_expected = np.asarray([[[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]],
                                  [[0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]]])

    assert data_utils.one_hot_encode(empty_test).size == 0
    assert np.allclose(data_utils.one_hot_encode(simple_test), simple_expected)

'''
def test_get_train_valid_indices(multi_task_lstm_crf_single_model, train_valid_indices_single_model):
    """Asserts that indices returned by the _get_train_valid_indices() of
    a MutliTaskLSTMCRf object are as expected."""
    # len of outer list
    assert len(train_valid_indices_single_model) == len(multi_task_lstm_crf_single_model.ds)
    # len of inner list
    assert len(train_valid_indices_single_model[0]) == multi_task_lstm_crf_single_model.config.k_folds
    # len of inner tuples
    assert len(train_valid_indices_single_model[0][0]) == 2

def test_get_data_partitions(multi_task_lstm_crf_single_model, data_partitions_single_model):
    """Asserts that partitions returned by the get_data_partitions() of
    a MutliTaskLSTMCRf object are as expected."""
    assert len(data_partitions_single_model) == len(multi_task_lstm_crf_single_model.ds)
    assert len(data_partitions_single_model[0]) == 6

def test_get_metrics(multi_task_lstm_crf_single_model, metrics_single_model):
    """Asserts that list of Metrics objects returned by get_metrics() is as expected."""
    ds_ = multi_task_lstm_crf_single_model.ds

    assert all(isinstance(m, Metrics) for m in metrics_single_model)
    assert isinstance(metrics_single_model, list)
    assert len(metrics_single_model) == len(ds_)
'''
