"""Any and all unit tests for the Embeddings class (saber/embeddings.py).
"""
import numpy as np

import pytest

from ..embeddings import Embeddings
from .resources.dummy_constants import (DUMMY_EMBEDDINGS_INDEX,
                                        DUMMY_EMBEDDINGS_MATRIX,
                                        DUMMY_TOKEN_MAP,
                                        PATH_TO_DUMMY_EMBEDDINGS)

# TODO (johngiorgi): write tests using a binary format file
# TODO (johngiorgi): write tests to test for debug functionality

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_embedding_idx():
    """Returns embedding index from call to `._prepare_embedding_index()`.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embedding_idx = embeddings._prepare_embedding_index(binary=False)
    return embedding_idx

@pytest.fixture
def dummy_embeddings_before_load():
    """Returns an instance of an Embeddings() object BEFORE the `.load()` method is called.
    """
    return Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                      token_map=DUMMY_TOKEN_MAP,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')

@pytest.fixture
def dummy_embeddings_after_load():
    """Returns an instance of an Embeddings() object AFTER the `.load()` method is called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, token_map=DUMMY_TOKEN_MAP)
    embeddings.load(binary=False) # txt file format is easier to test
    return embeddings

############################################ UNIT TESTS ############################################

def test_initialization(dummy_embeddings_before_load):
    """Asserts that Embeddings objects contain the expected attribute values after initialization.
    """
    # test attributes whos values are passed to the constructor
    assert dummy_embeddings_before_load.filepath == PATH_TO_DUMMY_EMBEDDINGS
    assert dummy_embeddings_before_load.token_map == DUMMY_TOKEN_MAP
    # test attributes initialized with default values
    assert dummy_embeddings_before_load.matrix is None
    assert dummy_embeddings_before_load.word_count is None
    assert dummy_embeddings_before_load.dimension is None
    # test that we can pass arbitrary keyword arguments
    assert dummy_embeddings_before_load.totally_arbitrary == 'arbitrary'

def test_prepare_embedding_index(dummy_embedding_idx):
    """Asserts that we get the expected value back after call to `._prepare_embedding_index()`.
    """
    # need to check keys and values differently
    assert dummy_embedding_idx.keys() == DUMMY_EMBEDDINGS_INDEX.keys()
    assert all(np.allclose(actual, expected) for actual, expected in
               zip(dummy_embedding_idx.values(), DUMMY_EMBEDDINGS_INDEX.values()))

def test_prepare_embedding_matrix(dummy_embedding_idx, dummy_embeddings_after_load):
    """Asserts that we get the expected value back after successive calls to
    `._prepare_embedding_index()` and `._prepare_embedding_matrix`.
    """
    embedding_matrix = dummy_embeddings_after_load._prepare_embedding_matrix(dummy_embedding_idx)
    assert np.allclose(embedding_matrix, DUMMY_EMBEDDINGS_MATRIX)

def test_matrix_after_load(dummy_embeddings_after_load):
    """Asserts that pre-trained token embeddings are loaded correctly when Embeddings.load() is
    called."""
    assert np.allclose(dummy_embeddings_after_load.matrix, DUMMY_EMBEDDINGS_MATRIX)

def test_attributes_after_load(dummy_embeddings_after_load):
    """Asserts that attributes of Embeddings object are updated as expected after `.load()` is
    called.
    """
    word_count, dimension = dummy_embeddings_after_load.matrix.shape
    assert dummy_embeddings_after_load.word_count == word_count
    assert dummy_embeddings_after_load.dimension == dimension
