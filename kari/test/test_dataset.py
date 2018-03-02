import pytest
import numpy
import pandas

from dataset import Dataset

# collect helpful articles here:
# https://semaphoreci.com/community/tutorials/testing-python-applications-with-pytest

# TODO (johngiorgi): is there a better way to get the dummy_word_types?
# TODO (johngiorgi): Update example with words 'null' and '"', as these
# gave me a lot of trouble.
# TODO (johngiorgi): Need tests for compound datasets

# constants for dummy dataset to perform testing on
# for the purposes of testing, test.tsv is EMPTY
DUMMY_WORD_TYPES = ["Human", "APC2", "maps", "to", "chromosome", "19p13",
".", "The", "absence", "of", "functional", "C7", "activity", "could", "not", "be",
"accounted", "for", "on", "the", "basis", "an", "inhibitor", "ENDPAD"]
DUMMY_TAG_TYPES = ["O", "B-Disease", "I-Disease", "E-Disease"]
DUMMY_TRAIN_SENT = [[("Human", "O"), ("APC2", "O"), ("maps", "O"), ("to", "O"),
("chromosome", "O"), ("19p13", "O"), (".", "O")], [("The", "O"),
("absence", "B-Disease"), ("of", "I-Disease"), ("functional", "I-Disease"), ("C7", "E-Disease"),
("activity", "O"), ("could", "O"), ("not", "O"), ("be", "O"), ("accounted", "O"),
("for", "O"), ("on", "O"), ("the", "O"), ("basis", "O"), ("of", "O"),
("an", "O"), ("inhibitor", "O"), (".", "O")]]
PATH_TO_DUMMY_DATASET = 'kari/test/resources/dummy_dataset'

@pytest.fixture
def empty_dummy_dataset():
    """ Returns an empty 'dummy' Dataset instance."""
    dataset = Dataset(PATH_TO_DUMMY_DATASET, sep='\t')

    return dataset

@pytest.fixture
def dummy_dataset():
    """ Returns a 'dummy' Dataset instance with two sentences."""
    dataset = Dataset(PATH_TO_DUMMY_DATASET, sep='\t')
    dataset.load_dataset()

    return dataset

def test_attributes_after_initilization_of_dataset(empty_dummy_dataset):
    """ Asserts instance attributes are initialized correctly when dataset is
    empty (i.e., load_dataset() method has not been called.)"""
    # attributes that are passed to __init__
    assert type(empty_dummy_dataset.dataset_folder) == str
    assert type(empty_dummy_dataset.trainset_filepath) == str
    # assert type(empty_dummy_dataset.testset_filepath) == str
    assert type(empty_dummy_dataset.sep) == str
    assert type(empty_dummy_dataset.header) == bool or int or list
    assert type(empty_dummy_dataset.names) == bool or list
    assert type(empty_dummy_dataset.max_seq_len) == int
    # other instance attributes
    # ensure we get the expected types
    assert type(empty_dummy_dataset.raw_dataframe) == pandas.DataFrame
    assert type(empty_dummy_dataset.word_types) == list
    assert type(empty_dummy_dataset.tag_types) == list
    assert type(empty_dummy_dataset.word_type_count) == int
    assert type(empty_dummy_dataset.tag_type_count) == int
    # assert type(empty_dummy_dataset.train_dataframe) == pandas.DataFrame
    # assert type(empty_dummy_dataset.test_dataframe) == pandas.DataFrame
    # ensure we get the expected values
    assert set(empty_dummy_dataset.word_types) == set(DUMMY_WORD_TYPES)
    assert set(empty_dummy_dataset.tag_types) == set(DUMMY_TAG_TYPES)
    assert empty_dummy_dataset.word_type_count == len(DUMMY_WORD_TYPES)
    assert empty_dummy_dataset.tag_type_count == len(DUMMY_TAG_TYPES)
    assert empty_dummy_dataset.word_type_to_idx == {}
    assert empty_dummy_dataset.tag_type_to_idx == {}
    assert empty_dummy_dataset.train_sentences == []
    assert empty_dummy_dataset.train_word_idx_sequence == []
    assert empty_dummy_dataset.train_tag_idx_sequence == []
    # assert empty_dummy_dataset.test_sentences == []
    # assert empty_dummy_dataset.test_word_idx_sequence == []
    # assert empty_dummy_dataset.test_tag_idx_sequence == []

def test_type_to_idx_after_dataset_loaded(dummy_dataset):
    """ Asserts that word_type_to_idx and tag_type_to_idx are updated as
    expected after a call to load_dataset()."""
    # ensure we get the expected types after dataset is loaded
    assert type(dummy_dataset.word_type_to_idx) == dict
    assert type(dummy_dataset.tag_type_to_idx) == dict
    # ensure that type to index mapping is of expected length
    assert len(dummy_dataset.word_type_to_idx) == len(DUMMY_WORD_TYPES)
    assert len(dummy_dataset.tag_type_to_idx) == len(DUMMY_TAG_TYPES)
    # ensure that type to index mapping contains the expected keys
    assert all(key in DUMMY_WORD_TYPES for key in
               dummy_dataset.word_type_to_idx.keys())
    assert all(key in DUMMY_TAG_TYPES for key in
               dummy_dataset.tag_type_to_idx.keys())

def test_sentences_after_dataset_loaded(dummy_dataset):
    """ Asserts that sentences are updated as expected after a call to
    load_dataset()."""
    # ensure we get the expected type after dataset is loaded
    assert type(dummy_dataset.train_sentences) == list
    # assert type(dummy_dataset.test_sentences) == list
    # ensure that sentences contain the expected values
    assert dummy_dataset.train_sentences == DUMMY_TRAIN_SENT
    # assert dummy_dataset.test_sentences == DUMMY_TEST_SENT

def test_type_idx_sequence_after_dataset_loaded(dummy_dataset):
    """ Asserts that word_idx_sequence and tag_idx_sequence are updated as
    expected after a call to load_dataset()."""
    # ensure we get the expected type after dataset is loaded
    # train
    assert type(dummy_dataset.train_word_idx_sequence) == numpy.ndarray
    assert type(dummy_dataset.train_tag_idx_sequence) == numpy.ndarray
    # test
    # assert type(dummy_dataset.test_word_idx_sequence) == numpy.ndarray
    # assert type(dummy_dataset.test_tag_idx_sequence) == numpy.ndarray

    # ensure that sentences contain the expected values
    # train
    assert dummy_dataset.train_word_idx_sequence.shape == (len(DUMMY_TRAIN_SENT),
                                                           dummy_dataset.max_seq_len)
    assert dummy_dataset.train_tag_idx_sequence.shape == (len(DUMMY_TRAIN_SENT),
                                                          dummy_dataset.max_seq_len,
                                                          len(DUMMY_TAG_TYPES))
    # test
    # assert dummy_dataset.test_word_idx_sequence.shape == (len(DUMMY_TEST_SENT),
    #                                                       dummy_dataset.max_seq_len)
    # assert dummy_dataset.test_tag_idx_sequence.shape == (len(DUMMY_TEST_SENT),
    #                                                      dummy_dataset.max_seq_len,
    #                                                      len(DUMMY_TAG_TYPES))
