import os
import numpy
import pytest
import pandas

from dataset import Dataset

# collect helpful articles here:
# https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest

# TODO (johngiorgi): is there a better way to get the dummy_word_types?
# TODO (johngiorgi): Update example with words 'null' and '"', as these
# gave me a lot of trouble.
# TODO (johngiorgi): need to write tests for compound dataset

# constants for dummy dataset to perform testing on
DUMMY_WORD_TYPES = ["Human", "APC2", "maps", "to", "chromosome", "19p13",
".", "The", "absence", "of", "functional", "C7", "activity", "could", "not", "be",
"accounted", "for", "on", "the", "basis", "an", "inhibitor", "ENDPAD"]
DUMMY_CHAR_TYPES = ['2', 's', 'E', 'c', 'N', 'T', 'd', 'e', 'H', 'h', 'a', 'b',
'v', 'C', 'm', 't', '9', 'P', 'p', 'r', '3', 'D', 'u', '.', 'o', '7', 'n', 'f',
'y', 'l', '1', 'i', 'A']
DUMMY_TAG_TYPES = ["O", "B-DISO", "I-DISO", "E-DISO"]
DUMMY_TRAIN_SENT = [[("Human", "O"), ("APC2", "O"), ("maps", "O"), ("to", "O"),
("chromosome", "O"), ("19p13", "O"), (".", "O")], [("The", "O"),
("absence", "B-DISO"), ("of", "I-DISO"), ("functional", "I-DISO"), ("C7", "E-DISO"),
("activity", "O"), ("could", "O"), ("not", "O"), ("be", "O"), ("accounted", "O"),
("for", "O"), ("on", "O"), ("the", "O"), ("basis", "O"), ("of", "O"),
("an", "O"), ("inhibitor", "O"), (".", "O")]]
PATH_TO_DUMMY_DATASET = 'saber/test/resources/single_dummy_dataset'

@pytest.fixture
def empty_dummy_dataset():
    """Returns an empty dummy Dataset instance."""
    dataset = Dataset(PATH_TO_DUMMY_DATASET)

    return dataset

@pytest.fixture
def single_dummy_dataset():
    """Returns a 'single' dummy Dataset instance."""
    dataset = Dataset(PATH_TO_DUMMY_DATASET)
    dataset.load_dataset()

    return dataset

@pytest.fixture
def compound_dummy_dataset():
    """Returns a 'compound' dummy Dataset instance."""
    dataset = Dataset(PATH_TO_DUMMY_DATASET)
    # need to passs shared word and shared char to idx
    dataset.load_dataset()

    return dataset

def test_attributes_after_initilization_of_dataset(empty_dummy_dataset):
    """Asserts instance attributes are initialized correctly when dataset is
    empty (i.e., load_dataset() method has not been called.)"""
    # attributes that are passed to __init__
    assert empty_dummy_dataset.dataset_folder == PATH_TO_DUMMY_DATASET
    assert empty_dummy_dataset.trainset_filepath == os.path.join(PATH_TO_DUMMY_DATASET, 'train.tsv')
    assert empty_dummy_dataset.sep == '\t'
    assert empty_dummy_dataset.names == ['Word', 'Tag']
    assert empty_dummy_dataset.header == None
    assert empty_dummy_dataset.max_seq_len == 75
    # other instance attributes
    assert empty_dummy_dataset.word_type_to_idx == {}
    assert empty_dummy_dataset.char_type_to_idx == {}
    assert empty_dummy_dataset.tag_type_to_idx == {}

    assert type(empty_dummy_dataset.raw_dataframe) == pandas.DataFrame
    assert type(empty_dummy_dataset.word_types) == list
    assert set(empty_dummy_dataset.word_types) == set(DUMMY_WORD_TYPES)
    assert type(empty_dummy_dataset.char_types) == list
    assert set(empty_dummy_dataset.char_types) == set(DUMMY_CHAR_TYPES)
    assert type(empty_dummy_dataset.tag_types) == list
    assert set(empty_dummy_dataset.tag_types) == set(DUMMY_TAG_TYPES)

    assert empty_dummy_dataset.word_type_count == len(DUMMY_WORD_TYPES)
    assert empty_dummy_dataset.char_type_count == len(DUMMY_CHAR_TYPES)
    assert empty_dummy_dataset.tag_type_count == len(DUMMY_TAG_TYPES)

    assert empty_dummy_dataset.train_sentences == []
    assert empty_dummy_dataset.train_word_idx_sequence == []
    assert empty_dummy_dataset.train_tag_idx_sequence == []

def test_type_to_idx_after_dataset_loaded(single_dummy_dataset):
    """Asserts that word_type_to_idx and tag_type_to_idx are updated as
    expected after a call to load_dataset()."""
    # ensure we get the expected types after dataset is loaded
    assert type(single_dummy_dataset.word_type_to_idx) == dict
    assert type(single_dummy_dataset.char_type_to_idx) == dict
    assert type(single_dummy_dataset.tag_type_to_idx) == dict
    # ensure that type to index mapping is of expected length
    assert len(single_dummy_dataset.word_type_to_idx) == len(DUMMY_WORD_TYPES)
    assert len(single_dummy_dataset.char_type_to_idx) == len(DUMMY_CHAR_TYPES)
    assert len(single_dummy_dataset.tag_type_to_idx) == len(DUMMY_TAG_TYPES)
    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in
               single_dummy_dataset.word_type_to_idx.keys())
    assert all(key in DUMMY_CHAR_TYPES for key in
               single_dummy_dataset.char_type_to_idx.keys())
    assert all(key in DUMMY_TAG_TYPES for key in
               single_dummy_dataset.tag_type_to_idx.keys())

def test_sentences_after_dataset_loaded(single_dummy_dataset):
    """Asserts that sentences are updated as expected after a call to
    load_dataset()."""
    assert single_dummy_dataset.train_sentences == DUMMY_TRAIN_SENT

def test_type_idx_sequence_after_dataset_loaded(single_dummy_dataset):
    """ Asserts that word_idx_sequence and tag_idx_sequence are updated as
    expected after a call to load_dataset()."""
    # ensure we get the expected type after dataset is loaded
    assert type(single_dummy_dataset.train_word_idx_sequence) == numpy.ndarray
    assert type(single_dummy_dataset.train_tag_idx_sequence) == numpy.ndarray

    # ensure that sentences are of the expected length
    assert single_dummy_dataset.train_word_idx_sequence.shape == (
        len(DUMMY_TRAIN_SENT), single_dummy_dataset.max_seq_len
    )
    assert single_dummy_dataset.train_tag_idx_sequence.shape == (
        len(DUMMY_TRAIN_SENT), single_dummy_dataset.max_seq_len, len(DUMMY_TAG_TYPES)
    )
