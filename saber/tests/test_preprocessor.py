"""Contains any and all unit tests for the `Preprocessor` class (saber/preprocessor.py).
"""
import pytest

from .. import constants
from ..preprocessor import Preprocessor


class TestPreprocessor(object):
    """Collects all unit tests for `saber.preprocessor.Preprocessor`.
    """
    def test_tokenize(self, preprocessor, nlp):
        """Asserts that call to Preprocessor._process_text() returns the expected
        results."""
        # simple test and its expected value
        simple_text = nlp("Simple example. With two sentences!")
        simple_expected = ([['Simple', 'example', '.'], ['With', 'two', 'sentences', '!']],
                           [[(0, 6), (7, 14), (14, 15)], [(16, 20), (21, 24), (25, 34), (34, 35)]])
        # blank value test and its expected value
        blank_test = nlp("")
        blank_expected = ([], [])

        assert preprocessor.tokenize(simple_text) == simple_expected
        assert preprocessor.tokenize(blank_test) == blank_expected

    def test_type_to_idx_value_error(self):
        """Asserts that `type_to_idx`()` returns a `ValueError` when the values of the
        `initial_mapping` argument do not form a contagious seqence.
        """
        with pytest.raises(ValueError):
            invalid_input = {'a': 0, 'b': 2, 'c': 3}
            Preprocessor.type_to_idx([], initial_mapping=invalid_input)

    def test_type_to_idx_empty_input(self):
        """Asserts that call to Preprocessor.test_type_to_idx() returns the expected results when
        an empty list is passed as input."""
        expected = {}
        actual = Preprocessor.type_to_idx([])

        assert actual == expected

    def test_type_to_idx_simple_input(self):
        """Asserts that call to Preprocessor.test_type_to_idx() returns the expected results when
        a simple list of strings is passed as input."""
        test = ["This", "is", "a", "test", "."]
        expected = {'This': 0, 'is': 1, 'a': 2, 'test': 3, '.': 4}
        actual = Preprocessor.type_to_idx(test)

        assert actual == expected

    def test_type_to_idx_intial_mapping(self):
        """Asserts that call to Preprocessor.test_type_to_idx() returns the expected results when
        a simple list of strings is passed as input and a supplied `intitial_mapping` argument"""
        test = ["This", "is", "a", "test", "."]
        initial_mapping = {'This': 0, 'is': 1}

        expected = {'This': 0, 'is': 1, 'a': 2, 'test': 3, '.': 4}
        actual = Preprocessor.type_to_idx(test, initial_mapping=initial_mapping)

        assert actual == expected

    def test_get_type_to_idx_sequence(self):
        """"""
        simple_seq = ["This", "is", "a", "test", ".", constants.UNK]
        simple_type_to_idx = Preprocessor.type_to_idx(simple_seq)
        simple_expected = [0, 1, 2, 3, 4]
        simple_actual = \
            Preprocessor.get_type_idx_sequence(simple_seq, type_to_idx=simple_type_to_idx)

        pass

    def test_sterilize(self):
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
