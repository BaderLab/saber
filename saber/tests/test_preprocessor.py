import en_coref_md
import pytest

from .. import constants
from ..preprocessor import Preprocessor

@pytest.fixture
def preprocessor():
    """Returns an instance of a Preprocessor object."""
    return Preprocessor()

@pytest.fixture
def nlp():
    """Returns Sacy NLP model."""
    return en_coref_md.load()

def test_process_text(preprocessor, nlp):
    """Asserts that call to Preprocessor._process_text() returns the expected
    results."""
    # simple test and its expected value
    simple_text = nlp("Simple example. With two sentences!")
    simple_expected = ([['Simple', 'example', '.'], ['With', 'two', \
        'sentences', '!']], [[(0, 6), (7, 14), (14, 15)], [(16, 20),\
         (21, 24), (25, 34), (34, 35)]])
    # blank value test and its expected value
    blank_test = nlp("")
    blank_expected = ([], [])

    assert preprocessor._process_text(simple_text) == simple_expected
    assert preprocessor._process_text(blank_test) == blank_expected

def test_type_to_idx():
    """Asserts that call to Preprocessor.test_type_to_idx() returns the
    expected results."""
    # simple test and its expected value
    simple_seq = ["This", "is", "a", "test", "."]
    simple_expected = {'This': 0, 'is': 1, 'a': 2, 'test': 3, '.': 4}
    # result of simple test with an offset arg value of 1
    offset_expected = {'This': 1, 'is': 2, 'a': 3, 'test': 4, '.': 5}
    # blank value test and its expected value
    blank_seq = []
    blank_expected = {}

    assert Preprocessor.type_to_idx(simple_seq) == simple_expected
    assert Preprocessor.type_to_idx(simple_seq, offset=1) == offset_expected
    assert Preprocessor.type_to_idx(blank_seq) == blank_expected

def test_get_type_to_idx_sequence():
    """"""
    simple_seq = ["This", "is", "a", "test", ".", constants.UNK]
    simple_type_to_idx = Preprocessor.type_to_idx(simple_seq)
    simple_expected = [0, 1, 2, 3, 4]
    simple_actual = Preprocessor.get_type_idx_sequence(seq=simple_seq, type_to_idx=simple_type_to_idx)
    # TODO

def test_chunk_entities():
    """Asserts that call to Preprocessor.chunk_entities() returns the
    expected results."""
    simple_seq = ['B-PRGE', 'I-PRGE', 'O', 'B-PRGE']
    simple_expected = [('PRGE', 0, 2), ('PRGE', 3, 4)]

    two_type_seq = ['B-LIVB', 'I-LIVB', 'O', 'B-PRGE']
    two_type_expected = [('LIVB', 0, 2), ('PRGE', 3, 4)]

    invalid_seq = ['O', 'I-CHED', 'I-CHED', 'O']
    invalid_expected = []

    blank_seq = []
    blank_expected = []

    assert Preprocessor.chunk_entities(simple_seq) == simple_expected
    assert Preprocessor.chunk_entities(two_type_seq) == two_type_expected
    assert Preprocessor.chunk_entities(invalid_seq) == invalid_expected
    assert Preprocessor.chunk_entities(blank_seq) == blank_expected

def test_sterilize():
    """Asserts that call to Preprocessor.sterilize() returns the
    expected results."""
    # test for proceeding and preeceding spaces
    simple_text = " This is an easy test. "
    simple_expected = "This is an easy test."
    # test for mutliple inline spacing errors
    multiple_spaces_text = "This  is a test   with improper  spacing. "
    multiple_spaces_expected = "This is a test with improper spacing."
    # blank value test and its expected value
    blank_text = ""
    blank_expected = ""

    assert Preprocessor.sterilize(simple_text) == simple_expected
    assert Preprocessor.sterilize(multiple_spaces_text) == multiple_spaces_expected
    assert Preprocessor.sterilize(blank_text) == blank_expected
