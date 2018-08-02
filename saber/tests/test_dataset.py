# TODO (johngiorgi): Need to increase coverage to include cases where
# some combination of train, valid and test partitions are provided. Currently,
# only the case where a single partition (train.*) is provided is covered.

import os
import pytest
import numpy as np

from .. import constants
from ..dataset import Dataset

# constants for dummy dataset to perform testing on
PATH_TO_DUMMY_DATASET = os.path.abspath('saber/tests/resources/dummy_dataset_1')
DUMMY_WORD_SEQ = np.array([
    ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.'],
    ['The', 'absence', 'of', 'functional', 'C7', 'activity', 'could', 'not', 'be', 'accounted',
     'for', 'on', 'the', 'basis', 'of', 'an', 'inhibitor', '.'],
])
DUMMY_TAG_SEQ = np.array([
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'B-DISO', 'I-DISO', 'I-DISO', 'E-DISO', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
     'O', 'O', 'O'],
])
DUMMY_WORD_TYPES = ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.', 'The', 'absence',
                    'of', 'functional', 'C7', 'activity', 'could', 'not', 'be', 'accounted',
                    'for', 'on', 'the', 'basis', 'an', 'inhibitor']
DUMMY_CHAR_TYPES = ['2', 's', 'c', 'T', 'd', 'e', 'H', 'h', 'a', 'b', 'v', 'C', 'm', 't', '9', 'p',
                    'r', '3', 'u', '.', 'o', '7', 'n', 'f', 'y', 'l', '1', 'i', 'A', 'P']
DUMMY_TAG_TYPES = ['O', 'B-DISO', 'I-DISO', 'E-DISO']

@pytest.fixture
def empty_dummy_dataset():
    """Returns an empty single dummy Dataset instance"""
    # Don't replace rare tokens for the sake of testing
    return Dataset(PATH_TO_DUMMY_DATASET, replace_rare_tokens=False)

@pytest.fixture
def loaded_dummy_dataset():
    """Returns a single dummy Dataset instance after calling load_dataset()"""
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(PATH_TO_DUMMY_DATASET, replace_rare_tokens=False)
    dataset.load_dataset()

    return dataset

def test_attributes_after_initilization_of_dataset(empty_dummy_dataset):
    """Asserts instance attributes are initialized correctly when dataset is
    empty (i.e., load_dataset() method has not been called.)"""
    # attributes that are passed to __init__
    for partition in constants.PARTITIONS:
        filepath = os.path.join(PATH_TO_DUMMY_DATASET, '{}.tsv'.format(partition))
        assert empty_dummy_dataset.partition_filepaths[partition] == filepath

    assert empty_dummy_dataset.sep == '\t'
    assert not empty_dummy_dataset.replace_rare_tokens
    # other instance attributes
    assert empty_dummy_dataset.type_seq == \
        {'train': {'word': None, 'char': None, 'tag': None},
         'valid': {'word': None, 'char': None, 'tag': None},
         'test': {'word': None, 'char': None, 'tag': None}}
    assert empty_dummy_dataset.types == {'word': None, 'char': None, 'tag': None}
    assert empty_dummy_dataset.type_to_idx == {'word': None, 'char': None, 'tag': None}
    assert empty_dummy_dataset.idx_to_tag is None
    assert empty_dummy_dataset.idx_seq == \
        {'train': {'word': None, 'char': None, 'tag': None},
         'valid': {'word': None, 'char': None, 'tag': None},
         'test': {'word': None, 'char': None, 'tag': None}}

def test_load_data_and_labels(empty_dummy_dataset):
    """Asserts that word_seq and tag_seq are updated as expected after a call to
    load_data_and_labels()."""
    empty_dummy_dataset.load_data_and_labels()
    # ensure we get the expected sequences after load_data_and_labels is called
    assert np.array_equal(empty_dummy_dataset.type_seq['train']['word'], DUMMY_WORD_SEQ)
    assert np.array_equal(empty_dummy_dataset.type_seq['train']['tag'], DUMMY_TAG_SEQ)

def test_get_types(empty_dummy_dataset):
    """Asserts that word_types, char_types and tag_types are updated as expected
    after a call to load_data_and_labels() and get_types()."""
    empty_dummy_dataset.load_data_and_labels()
    empty_dummy_dataset.get_types()

    assert set(empty_dummy_dataset.types['word']) == set(DUMMY_WORD_TYPES)
    assert set(empty_dummy_dataset.types['char']) == set(DUMMY_CHAR_TYPES)
    assert set(empty_dummy_dataset.types['tag']) == set(DUMMY_TAG_TYPES)

def test_input_sequences_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that word_seq and tag_seq are updated as expected after a call to
    load_dataset() is made on a dataset."""
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['word'], DUMMY_WORD_SEQ)
    assert np.array_equal(loaded_dummy_dataset.type_seq['train']['tag'], DUMMY_TAG_SEQ)

def test_types_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that word_types, char_types and tag_types are updated as expected
    after a call to load_dataset() is made on a dataset."""
    assert set(loaded_dummy_dataset.types['word']) == set(DUMMY_WORD_TYPES)
    assert set(loaded_dummy_dataset.types['char']) == set(DUMMY_CHAR_TYPES)
    assert set(loaded_dummy_dataset.types['tag']) == set(DUMMY_TAG_TYPES)

def test_type_to_idx_mapping_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that word_type_to_idx, char_type_to_idx and tag_type_to_idx are
    updated as expected after a call to load_dataset() is made on a dataset."""
    # ensure that type to index mapping is of expected length
    # the + constant value accounts for special tokens
    assert len(loaded_dummy_dataset.type_to_idx['word']) == len(DUMMY_WORD_TYPES) + 2
    assert len(loaded_dummy_dataset.type_to_idx['char']) == len(DUMMY_CHAR_TYPES) + 2
    assert len(loaded_dummy_dataset.type_to_idx['tag']) == len(DUMMY_TAG_TYPES) + 1

    # ensure that type to index mapping contains the expected keys
    # TODO: Figure out how to perform this test
    '''assert all(key in DUMMY_WORD_TYPES_1 for key in
        loaded_dummy_dataset.word_type_to_idx.keys())
    assert all(key in DUMMY_CHAR_TYPES for key in
        loaded_dummy_dataset.char_type_to_idx.keys())
    assert all(key in DUMMY_TAG_TYPES for key in
        loaded_dummy_dataset.tag_type_to_idx.keys())'''

    # ensure that index mapping is a contigous sequence of numbers starting at 0
    # and ending at len(mapping) - 1 (inclusive)
    for i in range(len(loaded_dummy_dataset.type_to_idx['word'])):
        assert i in loaded_dummy_dataset.type_to_idx['word'].values()
    for i in range(len(loaded_dummy_dataset.type_to_idx['char'])):
        assert i in loaded_dummy_dataset.type_to_idx['char'].values()
    for i in range(len(loaded_dummy_dataset.type_to_idx['tag'])):
        assert i in loaded_dummy_dataset.type_to_idx['tag'].values()

    assert max(loaded_dummy_dataset.type_to_idx['word'].values()) == \
        len(loaded_dummy_dataset.type_to_idx['word']) - 1
    assert max(loaded_dummy_dataset.type_to_idx['char'].values()) == \
        len(loaded_dummy_dataset.type_to_idx['char']) - 1
    assert max(loaded_dummy_dataset.type_to_idx['tag'].values()) == \
        len(loaded_dummy_dataset.type_to_idx['tag']) - 1

    # assert special tokens are mapped to the correct indices
    assert loaded_dummy_dataset.type_to_idx['word'][constants.PAD] == 0
    assert loaded_dummy_dataset.type_to_idx['word'][constants.UNK] == 1
    assert loaded_dummy_dataset.type_to_idx['char'][constants.PAD] == 0
    assert loaded_dummy_dataset.type_to_idx['char'][constants.UNK] == 1
    assert loaded_dummy_dataset.type_to_idx['tag'][constants.PAD] == 0

def test_train_idx_sequences_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that train_word_idx_seq, train_char_idx_seq and train_tag_idx_seq
    are updated as expected after a call to load_dataset() is made on a dataset."""
    # ensure that type to index
    # ensure we get the expected type after dataset is loaded
    assert isinstance(loaded_dummy_dataset.idx_seq['train']['word'], np.ndarray)
    assert isinstance(loaded_dummy_dataset.idx_seq['train']['char'], np.ndarray)
    assert isinstance(loaded_dummy_dataset.idx_seq['train']['tag'], np.ndarray)

    # ensure that sentences are of the expected length]
    # plus one accounts for special PAD token
    assert loaded_dummy_dataset.idx_seq['train']['word'].shape[0] == len(DUMMY_WORD_SEQ)
    assert loaded_dummy_dataset.idx_seq['train']['char'].shape[0] == len(DUMMY_WORD_SEQ)
    assert loaded_dummy_dataset.idx_seq['train']['tag'].shape[0] == len(DUMMY_WORD_SEQ)
    assert loaded_dummy_dataset.idx_seq['train']['tag'].shape[-1] == len(DUMMY_TAG_TYPES) + 1
