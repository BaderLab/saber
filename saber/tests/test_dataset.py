"""Contains any and all unit tests for the `Dataset` class (saber/dataset.py).
"""
import os

import numpy as np
from nltk.corpus.reader.conll import ConllCorpusReader

import pytest

from .. import constants
from ..dataset import Dataset
from .resources.dummy_constants import *

# TODO (johngiorgi): Need to include tests for valid/test partitions
# TODO (johngiorgi): Need to include tests for compound datasets

######################################### PYTEST FIXTURES #########################################
@pytest.fixture
def empty_dummy_dataset():
    """Returns an empty single dummy Dataset instance.
    """
    # Don't replace rare tokens for the sake of testing
    return Dataset(directory=PATH_TO_DUMMY_DATASET, replace_rare=False,
                   # to test passing of arbitrary keyword args to constructor
                   totally_arbitrary='arbitrary')

@pytest.fixture
def loaded_dummy_dataset():
    """Returns a single dummy Dataset instance after calling Dataset.load().
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET, replace_rare=False)
    dataset.load()

    return dataset

############################################ UNIT TESTS ############################################

# SINGLE DATASET
# Tests on unloaded Dataset object (`Dataset.load()` was not called)

def test_attributes_after_initilization_of_dataset(empty_dummy_dataset):
    """Asserts instance attributes are initialized correctly when dataset is empty (i.e.,
    `Dataset.load()` has not been called).
    """
    # attributes that are passed to __init__
    for partition in empty_dummy_dataset.directory:
        expected = os.path.join(PATH_TO_DUMMY_DATASET, '{}.tsv'.format(partition))
        assert empty_dummy_dataset.directory[partition] == expected
    assert not empty_dummy_dataset.replace_rare
    # other instance attributes
    assert empty_dummy_dataset.conll_parser.root == PATH_TO_DUMMY_DATASET
    assert empty_dummy_dataset.types == {'word': None, 'char': None, 'tag': None}
    assert empty_dummy_dataset.type_seq == {'train': None, 'valid': None, 'test': None}
    assert empty_dummy_dataset.type_to_idx == {'word': None, 'char': None, 'tag': None}
    assert empty_dummy_dataset.idx_to_tag is None
    assert empty_dummy_dataset.idx_seq == {'train': None, 'valid': None, 'test': None}
    # test that we can pass arbitrary keyword arguments
    assert empty_dummy_dataset.totally_arbitrary == 'arbitrary'

def test_get_types_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.types` is updated as expected after call to `Dataset._get_types()`.
    """
    empty_dummy_dataset._get_types()

    assert all(x in DUMMY_WORD_TYPES for x in empty_dummy_dataset.types['word'])
    assert all(x in DUMMY_CHAR_TYPES for x in empty_dummy_dataset.types['char'])
    assert all(x in DUMMY_TAG_TYPES for x in empty_dummy_dataset.types['tag'])

def test_get_type_seq_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.type_seq` is updated as expected after call to
    `Dataset._get_type_seq()`.
    """
    empty_dummy_dataset._get_type_seq()

    assert np.array_equal(empty_dummy_dataset.type_seq['train']['word'], DUMMY_WORD_SEQ)
    assert np.array_equal(empty_dummy_dataset.type_seq['train']['char'], DUMMY_CHAR_SEQ)
    assert np.array_equal(empty_dummy_dataset.type_seq['train']['tag'], DUMMY_TAG_SEQ)

def test_get_idx_maps_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.type_to_idx` is updated as expected after successive calls to
    `Dataset._get_idx_maps()` and `Dataset._get_idx_maps()`.
    """
    empty_dummy_dataset._get_types()
    empty_dummy_dataset._get_idx_maps()

    # ensure that index mapping is a contigous sequence of numbers starting at 0
    assert all(v in range(0, len(DUMMY_WORD_TYPES)) for v in
               empty_dummy_dataset.type_to_idx['word'].values())
    assert all(v in range(0, len(DUMMY_CHAR_TYPES)) for v in
               empty_dummy_dataset.type_to_idx['char'].values())
    assert all(v in range(0, len(DUMMY_TAG_TYPES)) for v in
               empty_dummy_dataset.type_to_idx['tag'].values())
    # assert special tokens are mapped to the correct indices
    assert empty_dummy_dataset.type_to_idx['word'][constants.PAD] == 0
    assert empty_dummy_dataset.type_to_idx['word'][constants.UNK] == 1
    assert empty_dummy_dataset.type_to_idx['char'][constants.PAD] == 0
    assert empty_dummy_dataset.type_to_idx['char'][constants.UNK] == 1
    assert empty_dummy_dataset.type_to_idx['tag'][constants.PAD] == 0

    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in empty_dummy_dataset.type_to_idx['word'])
    assert all(key in DUMMY_CHAR_TYPES for key in empty_dummy_dataset.type_to_idx['char'])
    assert all(key in DUMMY_TAG_TYPES for key in empty_dummy_dataset.type_to_idx['tag'])

def test_get_idx_seq_before_load(empty_dummy_dataset):
    """Asserts that `Dataset.idx_seq` is updated as expected after successive calls to
    `Dataset._get_type_seq()`, `Dataset._get_idx_maps()` and `Dataset.get_idx_seq()`.
    """
    empty_dummy_dataset._get_types()
    empty_dummy_dataset._get_type_seq()
    empty_dummy_dataset._get_idx_maps()
    empty_dummy_dataset.get_idx_seq()

    # as a workaround to testing this directly, just check that shapes are as expected
    expected_word_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN)
    expected_char_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, constants.MAX_CHAR_LEN)
    expected_tag_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, len(DUMMY_TAG_TYPES))
    assert empty_dummy_dataset.idx_seq['train']['word'].shape == expected_word_idx_shape
    assert empty_dummy_dataset.idx_seq['train']['char'].shape == expected_char_idx_shape
    assert empty_dummy_dataset.idx_seq['train']['tag'].shape == expected_tag_idx_shape

# tests on loaded Dataset object (`Dataset.load()` was called)

def test_get_types_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.types` is updated as expected after call to `Dataset.load()`.
    """
    assert all(x in DUMMY_WORD_TYPES for x in loaded_dummy_dataset.types['word'])
    assert all(x in DUMMY_CHAR_TYPES for x in loaded_dummy_dataset.types['char'])
    assert all(x in DUMMY_TAG_TYPES for x in loaded_dummy_dataset.types['tag'])

def test_get_type_seq_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.type_seq` is updated as expected after call to `Dataset.load()`.
    """
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['word'], DUMMY_WORD_SEQ)
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['char'], DUMMY_CHAR_SEQ)
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['tag'], DUMMY_TAG_SEQ)

def test_get_idx_maps_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.type_to_idx` is updated as expected after call to `Dataset.load()`.
    """
    # ensure that index mapping is a contigous sequence of numbers starting at 0
    assert all(v in range(0, len(DUMMY_WORD_TYPES)) for v in
               loaded_dummy_dataset.type_to_idx['word'].values())
    assert all(v in range(0, len(DUMMY_CHAR_TYPES)) for v in
               loaded_dummy_dataset.type_to_idx['char'].values())
    assert all(v in range(0, len(DUMMY_TAG_TYPES)) for v in
               loaded_dummy_dataset.type_to_idx['tag'].values())
    # assert special tokens are mapped to the correct indices
    assert loaded_dummy_dataset.type_to_idx['word'][constants.PAD] == 0
    assert loaded_dummy_dataset.type_to_idx['word'][constants.UNK] == 1
    assert loaded_dummy_dataset.type_to_idx['char'][constants.PAD] == 0
    assert loaded_dummy_dataset.type_to_idx['char'][constants.UNK] == 1
    assert loaded_dummy_dataset.type_to_idx['tag'][constants.PAD] == 0

    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in loaded_dummy_dataset.type_to_idx['word'])
    assert all(key in DUMMY_CHAR_TYPES for key in loaded_dummy_dataset.type_to_idx['char'])
    assert all(key in DUMMY_TAG_TYPES for key in loaded_dummy_dataset.type_to_idx['tag'])

def test_get_idx_seq_after_load(loaded_dummy_dataset):
    """Asserts that `Dataset.idx_seq` is updated as expected after calls to `Dataset.load()`.
    """
    # as a workaround to testing this directly, just check that shapes are as expected
    expected_word_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN)
    expected_char_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, constants.MAX_CHAR_LEN)
    expected_tag_idx_shape = (len(DUMMY_WORD_SEQ), constants.MAX_SENT_LEN, len(DUMMY_TAG_TYPES))
    assert loaded_dummy_dataset.idx_seq['train']['word'].shape == expected_word_idx_shape
    assert loaded_dummy_dataset.idx_seq['train']['char'].shape == expected_char_idx_shape
    assert loaded_dummy_dataset.idx_seq['train']['tag'].shape == expected_tag_idx_shape

def test_value_error_load(empty_dummy_dataset):
    """Asserts that `Dataset.load()` raises a ValueError when `Dataset.directory` is None.
    """
    # Set directory to None to force error to arise
    empty_dummy_dataset.directory = None
    with pytest.raises(ValueError):
        empty_dummy_dataset.load()

# COMPOUND DATASET
