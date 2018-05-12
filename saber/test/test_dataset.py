import os
import pytest
import numpy as np

from dataset import Dataset

# constants for dummy dataset to perform testing on
PATH_TO_DUMMY_DATASET = 'saber/test/resources/dummy_dataset'
DUMMY_WORD_SEQ = [
    ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.'],
    ['The', 'absence', 'of', 'functional', 'C7', 'activity', 'could', 'not',
     'be', 'accounted', 'for', 'on', 'the', 'basis', 'of', 'an', 'inhibitor',
     '.']
    ]
DUMMY_WORD_SEQ = np.asarray(DUMMY_WORD_SEQ)
DUMMY_TAG_SEQ = [
    ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
    ['O', 'B-DISO', 'I-DISO', 'I-DISO', 'E-DISO', 'O', 'O', 'O', 'O', 'O', 'O',
    'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ]
DUMMY_TAG_SEQ = np.asarray(DUMMY_TAG_SEQ)
DUMMY_WORD_TYPES = ['Human', 'APC2', 'maps', 'to', 'chromosome', '19p13', '.',
'The', 'absence', 'of', 'functional', 'C7', 'activity', 'could', 'not', 'be',
'accounted', 'for', 'on', 'the', 'basis', 'an', 'inhibitor', '<PAD>', '<UNK>']
DUMMY_CHAR_TYPES = ['2', 's', 'c', 'T', 'd', 'e', 'H', 'h', 'a', 'b', 'v', 'C',
'm', 't', '9', 'p', 'r', '3', 'u', '.', 'o', '7', 'n', 'f', 'y', 'l', '1', 'i',
'A', 'P', '<PAD>', '<UNK>']
DUMMY_TAG_TYPES = ['O', 'B-DISO', 'I-DISO', 'E-DISO', '<PAD>']

@pytest.fixture
def empty_dummy_dataset():
    """Returns an empty dummy Dataset instance."""
    dataset = Dataset(PATH_TO_DUMMY_DATASET)

    return dataset

@pytest.fixture
def loaded_dummy_dataset():
    """Returns a dummy Dataset instance after calling load_dataset()"""
    dataset = Dataset(PATH_TO_DUMMY_DATASET)
    dataset.load_dataset()

    return dataset

def test_attributes_after_initilization_of_dataset(empty_dummy_dataset):
    """Asserts instance attributes are initialized correctly when dataset is
    empty (i.e., load_dataset() method has not been called.)"""
    # attributes that are passed to __init__
    assert empty_dummy_dataset.filepath == PATH_TO_DUMMY_DATASET
    trainset_filepath = os.path.join(PATH_TO_DUMMY_DATASET, 'train.tsv')
    assert empty_dummy_dataset.trainset_filepath == trainset_filepath
    assert empty_dummy_dataset.sep == '\t'
    assert empty_dummy_dataset.names == ['Word', 'Tag']
    assert empty_dummy_dataset.header == None
    # other instance attributes
    assert empty_dummy_dataset.word_seq == None
    assert empty_dummy_dataset.tag_seq == None

    assert empty_dummy_dataset.word_types == None
    assert empty_dummy_dataset.char_types == None
    assert empty_dummy_dataset.tag_types == None

    assert empty_dummy_dataset.word_type_to_idx == None
    assert empty_dummy_dataset.char_type_to_idx == None
    assert empty_dummy_dataset.tag_type_to_idx == None
    assert empty_dummy_dataset.idx_to_tag_type == None

    assert empty_dummy_dataset.train_word_idx_seq == None
    assert empty_dummy_dataset.train_char_idx_seq == None
    assert empty_dummy_dataset.train_tag_idx_seq == None

def test_load_data_and_labels(empty_dummy_dataset):
    """Asserts that word_seq and tag_seq are updated as expected after a call to
    load_data_and_labels()."""
    empty_dummy_dataset.load_data_and_labels()
    # ensure we get the expected sequences after load_data_and_labels is called
    assert np.array_equal(empty_dummy_dataset.word_seq, DUMMY_WORD_SEQ)
    assert np.array_equal(empty_dummy_dataset.tag_seq, DUMMY_TAG_SEQ)

def test_get_types(empty_dummy_dataset):
    """Asserts that word_types, char_types and tag_types are updated as expected
    after a call to load_data_and_labels() and get_types()."""
    empty_dummy_dataset.load_data_and_labels()
    empty_dummy_dataset.get_types()

    assert set(empty_dummy_dataset.word_types) == set(DUMMY_WORD_TYPES)
    assert set(empty_dummy_dataset.char_types) == set(DUMMY_CHAR_TYPES)
    assert set(empty_dummy_dataset.tag_types) == set(DUMMY_TAG_TYPES)

def test_map_type_to_idx(empty_dummy_dataset):
    """Asserts that word_type_to_idx, char_type_to_idx and tag_type_to_idx are
    updated as expected after a call to load_data_and_labels() and
    get_types()."""
    empty_dummy_dataset.load_data_and_labels()
    empty_dummy_dataset.get_types()
    empty_dummy_dataset._map_type_to_idx()

    # ensure that type to index mapping is of expected length
    assert len(empty_dummy_dataset.word_type_to_idx) == len(DUMMY_WORD_TYPES)
    assert len(empty_dummy_dataset.char_type_to_idx) == len(DUMMY_CHAR_TYPES)
    assert len(empty_dummy_dataset.tag_type_to_idx) == len(DUMMY_TAG_TYPES)
    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in
               empty_dummy_dataset.word_type_to_idx.keys())
    assert all(key in DUMMY_CHAR_TYPES for key in
               empty_dummy_dataset.char_type_to_idx.keys())
    assert all(key in DUMMY_TAG_TYPES for key in
               empty_dummy_dataset.tag_type_to_idx.keys())

def test_input_sequences_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that word_seq and tag_seq are updated as expected after a call to
    load_dataset() is made on a dataset."""
    assert np.array_equal(loaded_dummy_dataset.word_seq, DUMMY_WORD_SEQ)
    assert np.array_equal(loaded_dummy_dataset.tag_seq, DUMMY_TAG_SEQ)

def test_types_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that word_types, char_types and tag_types are updated as expected
    after a call to load_dataset() is made on a dataset."""
    assert set(loaded_dummy_dataset.word_types) == set(DUMMY_WORD_TYPES)
    assert set(loaded_dummy_dataset.char_types) == set(DUMMY_CHAR_TYPES)
    assert set(loaded_dummy_dataset.tag_types) == set(DUMMY_TAG_TYPES)

def test_type_to_idx_mapping_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that word_type_to_idx, char_type_to_idx and tag_type_to_idx are
    updated as expected after a call to load_dataset() is made on a dataset."""
    # ensure that type to index mapping is of expected length
    assert len(loaded_dummy_dataset.word_type_to_idx) == len(DUMMY_WORD_TYPES)
    assert len(loaded_dummy_dataset.char_type_to_idx) == len(DUMMY_CHAR_TYPES)
    assert len(loaded_dummy_dataset.tag_type_to_idx) == len(DUMMY_TAG_TYPES)
    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in
               loaded_dummy_dataset.word_type_to_idx.keys())
    assert all(key in DUMMY_CHAR_TYPES for key in
               loaded_dummy_dataset.char_type_to_idx.keys())
    assert all(key in DUMMY_TAG_TYPES for key in
               loaded_dummy_dataset.tag_type_to_idx.keys())

def test_train_idx_sequences_after_loading_dataset(loaded_dummy_dataset):
    """Asserts that train_word_idx_seq, train_char_idx_seq and train_tag_idx_seq
    are updated as expected after a call to load_dataset() is made on a dataset."""
    # ensure that type to index
    # ensure we get the expected type after dataset is loaded
    assert type(loaded_dummy_dataset.train_word_idx_seq) == np.ndarray
    assert type(loaded_dummy_dataset.train_char_idx_seq) == np.ndarray
    assert type(loaded_dummy_dataset.train_tag_idx_seq) == np.ndarray

    # ensure that sentences are of the expected length
    assert loaded_dummy_dataset.train_word_idx_seq.shape[0] == len(DUMMY_WORD_SEQ)
    assert loaded_dummy_dataset.train_char_idx_seq.shape[0] == len(DUMMY_WORD_SEQ)
    assert loaded_dummy_dataset.train_tag_idx_seq.shape[0] == len(DUMMY_WORD_SEQ)
    assert loaded_dummy_dataset.train_tag_idx_seq.shape[-1] == len(DUMMY_TAG_TYPES)
