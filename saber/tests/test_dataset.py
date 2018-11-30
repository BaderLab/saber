"""Contains any and all unit tests for the `Dataset` class (saber/dataset.py).
"""
import os

import numpy as np
from nltk.corpus.reader.conll import ConllCorpusReader

import pytest

from .. import constants
from ..dataset import Dataset
from ..utils import generic_utils
from .resources.dummy_constants import *

# TODO (johngiorgi): Need to include tests for valid/test partitions
# TODO (johngiorgi): Need to include tests for compound datasets

######################################### PYTEST FIXTURES #########################################
@pytest.fixture
def empty_dummy_dataset():
    """Returns an empty single dummy Dataset instance.
    """
    # Don't replace rare tokens for the sake of testing
    return Dataset(directory=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False,
                   # to test passing of arbitrary keyword args to constructor
                   totally_arbitrary='arbitrary')

@pytest.fixture
def loaded_dummy_dataset():
    """Returns a single dummy Dataset instance after calling Dataset.load().
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False)
    dataset.load()

    return dataset

############################################ UNIT TESTS ############################################

# Generic object tests

def test_attributes_after_initilization_of_dataset(empty_dummy_dataset):
    """Asserts instance attributes are initialized correctly when dataset is empty (i.e.,
    `Dataset.load()` has not been called).
    """
    # attributes that are passed to __init__
    for partition in empty_dummy_dataset.directory:
        expected = os.path.join(PATH_TO_DUMMY_DATASET_1, '{}.tsv'.format(partition))
        assert empty_dummy_dataset.directory[partition] == expected
    assert not empty_dummy_dataset.replace_rare_tokens
    # other instance attributes
    assert empty_dummy_dataset.conll_parser.root == PATH_TO_DUMMY_DATASET_1
    assert empty_dummy_dataset.type_seq == {'train': None, 'valid': None, 'test': None}
    assert empty_dummy_dataset.type_to_idx == {'word': None, 'char': None, 'tag': None}
    assert empty_dummy_dataset.idx_to_tag is None
    assert empty_dummy_dataset.idx_seq == {'train': None, 'valid': None, 'test': None}
    # test that we can pass arbitrary keyword arguments
    assert empty_dummy_dataset.totally_arbitrary == 'arbitrary'

def test_value_error_load(empty_dummy_dataset):
    """Asserts that `Dataset.load()` raises a ValueError when `Dataset.directory` is None.
    """
    # Set directory to None to force error to arise
    empty_dummy_dataset.directory = None
    with pytest.raises(ValueError):
        empty_dummy_dataset.load()

# SINGLE DATASET

def test_get_types_single_dataset(empty_dummy_dataset):
    """Asserts that `Dataset._get_types()` returns the expected values.
    """
    actual = empty_dummy_dataset._get_types()
    expected = {'word': DUMMY_WORD_TYPES, 'char': DUMMY_CHAR_TYPES, 'tag': DUMMY_TAG_TYPES}

    # sort allows us to assert that the two lists are identical
    assert all(actual['word'].sort() == v.sort() for k, v in expected.items())

# Tests on unloaded Dataset object (`Dataset.load()` was not called)

def test_get_type_seq_single_dataset_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.type_seq` is updated as expected after call to
    `Dataset._get_type_seq()`.
    """
    empty_dummy_dataset._get_type_seq()

    assert np.array_equal(empty_dummy_dataset.type_seq['train']['word'], DUMMY_WORD_SEQ)
    assert np.array_equal(empty_dummy_dataset.type_seq['train']['char'], DUMMY_CHAR_SEQ)
    assert np.array_equal(empty_dummy_dataset.type_seq['train']['tag'], DUMMY_TAG_SEQ)

def test_get_idx_maps_single_dataset_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.type_to_idx` is updated as expected after successive calls to
    `Dataset._get_types()` and `Dataset._get_idx_maps()`.
    """
    types = empty_dummy_dataset._get_types()
    empty_dummy_dataset._get_idx_maps(types)

    # ensure that index mapping is a contigous sequence of numbers starting at 0
    assert generic_utils.is_consecutive(empty_dummy_dataset.type_to_idx['word'].values())
    assert generic_utils.is_consecutive(empty_dummy_dataset.type_to_idx['char'].values())
    assert generic_utils.is_consecutive(empty_dummy_dataset.type_to_idx['tag'].values())
    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in empty_dummy_dataset.type_to_idx['word'])
    assert all(key in DUMMY_CHAR_TYPES for key in empty_dummy_dataset.type_to_idx['char'])
    assert all(key in DUMMY_TAG_TYPES for key in empty_dummy_dataset.type_to_idx['tag'])

def test_get_idx_maps_single_dataset_before_load_special_tokens(empty_dummy_dataset):
    """Asserts that `Dataset.type_to_idx` contains the special tokens as keys with expected values
     after successive calls to `Dataset._get_types()` and `Dataset._get_idx_maps()`.
    """
    types = empty_dummy_dataset._get_types()
    empty_dummy_dataset._get_idx_maps(types)
    # assert special tokens are mapped to the correct indices
    assert all(empty_dummy_dataset.type_to_idx['word'][k] == v for k, v in constants.INITIAL_MAPPING['word'].items())
    assert all(empty_dummy_dataset.type_to_idx['char'][k] == v for k, v in constants.INITIAL_MAPPING['word'].items())
    assert all(empty_dummy_dataset.type_to_idx['tag'][k] == v for k, v in constants.INITIAL_MAPPING['tag'].items())

def test_get_idx_seq_single_dataset_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.idx_seq` is updated as expected after successive calls to
    `Dataset._get_type_seq()`, `Dataset._get_idx_maps()` and `Dataset.get_idx_seq()`.
    """
    types = empty_dummy_dataset._get_types()
    empty_dummy_dataset._get_type_seq()
    empty_dummy_dataset._get_idx_maps(types)
    empty_dummy_dataset.get_idx_seq()

    # as a workaround to testing this directly, just check that shapes are as expected
    expected_word_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN)
    expected_char_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, constants.MAX_CHAR_LEN)
    expected_tag_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, len(DUMMY_TAG_TYPES))

    assert all(empty_dummy_dataset.idx_seq[partition]['word'].shape == expected_word_idx_shape
               for partition in ['train', 'test', 'valid'])
    assert all(empty_dummy_dataset.idx_seq[partition]['char'].shape == expected_char_idx_shape
               for partition in ['train', 'test', 'valid'])
    assert all(empty_dummy_dataset.idx_seq[partition]['tag'].shape == expected_tag_idx_shape
               for partition in ['train', 'test', 'valid'])

# tests on loaded Dataset object (`Dataset.load()` was called)

def test_get_type_seq_single_dataset_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.type_seq` is updated as expected after call to `Dataset.load()`.
    """
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['word'], DUMMY_WORD_SEQ)
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['char'], DUMMY_CHAR_SEQ)
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['tag'], DUMMY_TAG_SEQ)

def test_get_idx_maps_single_dataset_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.type_to_idx` is updated as expected after call to `Dataset.load()`.
    """
    # ensure that index mapping is a contigous sequence of numbers starting at 0
    # ensure that index mapping is a contigous sequence of numbers starting at 0
    assert generic_utils.is_consecutive(loaded_dummy_dataset.type_to_idx['word'].values())
    assert generic_utils.is_consecutive(loaded_dummy_dataset.type_to_idx['char'].values())
    assert generic_utils.is_consecutive(loaded_dummy_dataset.type_to_idx['tag'].values())
    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in loaded_dummy_dataset.type_to_idx['word'])
    assert all(key in DUMMY_CHAR_TYPES for key in loaded_dummy_dataset.type_to_idx['char'])
    assert all(key in DUMMY_TAG_TYPES for key in loaded_dummy_dataset.type_to_idx['tag'])

def test_get_idx_maps_single_dataset_after_load_special_tokens(loaded_dummy_dataset):
    """Asserts that `Dataset.type_to_idx` contains the special tokens as keys with expected values
     after successive calls to `Dataset._get_types()` and `Dataset.get_idx_maps()`.
    """
    # assert special tokens are mapped to the correct indices
    assert all(loaded_dummy_dataset.type_to_idx['word'][k] == v for k, v in constants.INITIAL_MAPPING['word'].items())
    assert all(loaded_dummy_dataset.type_to_idx['char'][k] == v for k, v in constants.INITIAL_MAPPING['word'].items())
    assert all(loaded_dummy_dataset.type_to_idx['tag'][k] == v for k, v in constants.INITIAL_MAPPING['tag'].items())

def test_get_idx_seq_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.idx_seq` is updated as expected after calls to `Dataset.load()`.
    """
    # as a workaround to testing this directly, just check that shapes are as expected
    expected_word_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN)
    expected_char_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, constants.MAX_CHAR_LEN)
    expected_tag_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, len(DUMMY_TAG_TYPES))

    assert all(loaded_dummy_dataset.idx_seq[partition]['word'].shape == expected_word_idx_shape
               for partition in ['train', 'test', 'valid'])
    assert all(loaded_dummy_dataset.idx_seq[partition]['char'].shape == expected_char_idx_shape
               for partition in ['train', 'test', 'valid'])
    assert all(loaded_dummy_dataset.idx_seq[partition]['tag'].shape == expected_tag_idx_shape
               for partition in ['train', 'test', 'valid'])

# COMPOUND DATASET
