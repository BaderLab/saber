"""Any and all unit tests for the Embeddings class (saber/embeddings.py).
"""
import numpy as np

import pytest

from ..embeddings import Embeddings
from .resources.dummy_constants import *
from .resources import helpers

# TODO (johngiorgi): write tests using a binary format file
# TODO (johngiorgi): write tests to test for debug functionality

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_embedding_idx():
    """Returns embedding index from call to `Embeddings._prepare_embedding_index()`.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    return embedding_idx

@pytest.fixture
def dummy_embedding_matrix_and_type_to_idx():
    """Returns the `embedding_matrix` and `type_to_index` objects from call to
    `Embeddings._prepare_embedding_matrix(load_all=False)`.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    embeddings.num_found = len(embedding_idx)
    embeddings.dimension = len(list(embedding_idx.values())[0])
    embedding_matrix, type_to_idx = embeddings._prepare_embedding_matrix(embedding_idx, load_all=False)
    embeddings.num_embed = embedding_matrix.shape[0] # num of embedded words

    return embedding_matrix, type_to_idx

@pytest.fixture
def dummy_embedding_matrix_and_type_to_idx_load_all():
    """Returns the embedding matrix and type to index objects from call to
    `Embeddings._prepare_embedding_matrix(load_all=True)`.
    """
    # this should be different than DUMMY_TOKEN_MAP for a reliable test
    test = {"This": 0, "is": 1, "a": 2, "test": 3}

    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=test)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    embeddings.num_found = len(embedding_idx)
    embeddings.dimension = len(list(embedding_idx.values())[0])
    embedding_matrix, type_to_idx = embeddings._prepare_embedding_matrix(embedding_idx, load_all=True)
    embeddings.num_embed = embedding_matrix.shape[0] # num of embedded words

    return embedding_matrix, type_to_idx

@pytest.fixture
def dummy_embeddings_before_load():
    """Returns an instance of an Embeddings() object BEFORE the `Embeddings.load()` method is
    called.
    """
    return Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                      token_map=DUMMY_TOKEN_MAP,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')

@pytest.fixture
def dummy_embeddings_after_load():
    """Returns an instance of an Embeddings() object AFTER `Embeddings.load(load_all=False)` is
    called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embeddings.load(binary=False, load_all=False) # txt file format is easier to test
    return embeddings

@pytest.fixture
def dummy_embeddings_after_load_with_load_all():
    """Returns an instance of an Embeddings() object AFTER `Embeddings.load(load_all=True)` is
    called.
    """
    # this should be different than DUMMY_TOKEN_MAP for a reliable test
    test = {"This": 0, "is": 1, "a": 2, "test": 3}

    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=test)
    embeddings.load(binary=False, load_all=True) # txt file format is easier to test
    return embeddings

############################################ UNIT TESTS ############################################

def test_initialization(dummy_embeddings_before_load):
    """Asserts that Embeddings object contains the expected attribute values after initialization.
    """
    # test attributes whos values are passed to the constructor
    assert dummy_embeddings_before_load.filepath == PATH_TO_DUMMY_EMBEDDINGS
    assert dummy_embeddings_before_load.token_map == DUMMY_TOKEN_MAP
    # test attributes initialized with default values
    assert dummy_embeddings_before_load.matrix is None
    assert dummy_embeddings_before_load.num_found is None
    assert dummy_embeddings_before_load.num_embed is None
    assert dummy_embeddings_before_load.dimension is None
    # test that we can pass arbitrary keyword arguments
    assert dummy_embeddings_before_load.totally_arbitrary == 'arbitrary'

def test_prepare_embedding_index(dummy_embedding_idx):
    """Asserts that we get the expected value back after call to
    `Embeddings._prepare_embedding_index()`.
    """
    # need to check keys and values differently
    assert dummy_embedding_idx.keys() == DUMMY_EMBEDDINGS_INDEX.keys()
    assert all(np.allclose(actual, expected) for actual, expected in
               zip(dummy_embedding_idx.values(), DUMMY_EMBEDDINGS_INDEX.values()))

def test_prepare_embedding_matrix(dummy_embedding_matrix_and_type_to_idx):
    """Asserts that we get the expected value back after successive calls to
    `Embeddings._prepare_embedding_index()` and
    `Embeddings._prepare_embedding_matrix(load_all=False)`.
    """
    # expected_values
    embedding_matrix_expected, type_to_idx_expected = DUMMY_EMBEDDINGS_MATRIX, None
    # actual values
    embedding_matrix_actual, type_to_idx_actual = dummy_embedding_matrix_and_type_to_idx

    assert np.allclose(embedding_matrix_actual, embedding_matrix_expected)
    assert type_to_idx_actual is type_to_idx_expected

def test_prepare_embedding_matrix_load_all(dummy_embedding_matrix_and_type_to_idx_load_all):
    """Asserts that we get the expected value back after successive calls to
    `Embeddings._prepare_embedding_index()` and
    `Embeddings._prepare_embedding_matrix(load_all=True)`.
    """
    # expected values
    embedding_matrix_expected = DUMMY_EMBEDDINGS_MATRIX
    type_to_idx_expected = {"word": DUMMY_TOKEN_MAP, "char": DUMMY_CHAR_MAP}
    # actual values
    embedding_matrix_actual, type_to_idx_actual = dummy_embedding_matrix_and_type_to_idx_load_all

    assert np.allclose(embedding_matrix_actual, embedding_matrix_expected)
    helpers.assert_type_to_idx_as_expected(actual=type_to_idx_actual, expected=type_to_idx_expected)

def test_matrix_after_load(dummy_embeddings_after_load):
    """Asserts that pre-trained token embeddings are loaded correctly when
    `Embeddings.load(load_all=False)` is called."""
    assert np.allclose(dummy_embeddings_after_load.matrix, DUMMY_EMBEDDINGS_MATRIX)

def test_matrix_after_load_with_load_all(dummy_embeddings_after_load):
    """Asserts that pre-trained token embeddings are loaded correctly when
    `Embeddings.load(load_all=True)` is called."""
    assert np.allclose(dummy_embeddings_after_load.matrix, DUMMY_EMBEDDINGS_MATRIX)

def test_attributes_after_load(dummy_embedding_idx, dummy_embeddings_after_load):
    """Asserts that attributes of Embeddings object are updated as expected after
    `Embeddings.load(load_all=False)` is called.
    """
    # expected values
    num_found_expected = len(dummy_embedding_idx)
    dimension_expected = len(list(dummy_embedding_idx.values())[0])
    num_embed_expected = dummy_embeddings_after_load.matrix.shape[0]
    # actual values
    num_found_actual = dummy_embeddings_after_load.num_found
    dimension_actual = dummy_embeddings_after_load.dimension
    num_embed_actual = dummy_embeddings_after_load.num_embed

    assert num_found_expected == num_found_actual
    assert dimension_expected == dimension_actual
    assert num_embed_expected == num_embed_actual

def test_attributes_after_load_with_load_all(dummy_embedding_idx,
                                             dummy_embeddings_after_load_with_load_all):
    """Asserts that attributes of Embeddings object are updated as expected after
    `Embeddings.load(load_all=True)` is called.
    """
    # expected values
    num_found_expected = len(dummy_embedding_idx)
    dimension_expected = len(list(dummy_embedding_idx.values())[0])
    num_embed_expected = dummy_embeddings_after_load_with_load_all.matrix.shape[0]
    # actual values
    num_found_actual = dummy_embeddings_after_load_with_load_all.num_found
    dimension_actual = dummy_embeddings_after_load_with_load_all.dimension
    num_embed_actual = dummy_embeddings_after_load_with_load_all.num_embed

    assert num_found_expected == num_found_actual
    assert dimension_expected == dimension_actual
    assert num_embed_expected == num_embed_actual

def test_generate_type_to_idx(dummy_embeddings_before_load):
    """Asserts that the dictionary returned from 'Embeddings._generate_type_to_idx()' is as
    expected.
    """
    test = {'This': 0, 'is': 1, 'a': 2, 'test': 3}

    # expected values
    expected = {
        "word": list(test.keys()),
        "char": []
    }
    for word in expected['word']:
        expected['char'].extend(list(word))
    expected['char'] = list(set(expected['char']))
    # actual values
    actual = dummy_embeddings_before_load._generate_type_to_idx(test)

    helpers.assert_type_to_idx_as_expected(actual=actual, expected=expected)
