"""Test suite for the `Dataset` class (saber.dataset.Dataset).
"""
import os

import numpy as np
import pytest

from .. import constants
from ..constants import MAX_CHAR_LEN
from ..constants import MAX_SENT_LEN
from ..utils import generic_utils
from .resources.constants import PATH_TO_CONLL2003_DATASET
from .resources.constants import PATH_TO_CONLL2004_DATASET
from .resources.constants import CoNLL2003_CHAR_SEQ
from .resources.constants import CoNLL2003_CHAR_TYPES
from .resources.constants import CoNLL2003_ENT_SEQ
from .resources.constants import CoNLL2003_ENT_TYPES
from .resources.constants import CoNLL2003_WORD_SEQ
from .resources.constants import CoNLL2003_WORD_TYPES
from .resources.constants import CoNLL2004_CHAR_SEQ
from .resources.constants import CoNLL2004_CHAR_TYPES
from .resources.constants import CoNLL2004_ENT_SEQ
from .resources.constants import CoNLL2004_ENT_TYPES
from .resources.constants import CoNLL2004_REL_SEQ
from .resources.constants import CoNLL2004_REL_TYPES
from .resources.constants import CoNLL2004_WORD_SEQ
from .resources.constants import CoNLL2004_WORD_TYPES

# TODO (johngiorgi): Need to include tests for valid/test partitions
# TODO (johngiorgi): Need to include tests for compound datasets


class TestDataset(object):
    """Collects all unit tests for `saber.dataset.Dataset`.
    """
    def test_attributes_after_init_no_dataset_folder(self, dataset_no_dataset_folder):
        """Asserts that the attributes of a `Dataset` object are initialized with the expected
        values when no dataset folder is provided.
        """
        # Attributes that are passed to __init__
        assert dataset_no_dataset_folder.dataset_folder is None
        assert not dataset_no_dataset_folder.replace_rare_tokens

        # Other instance attributes
        assert dataset_no_dataset_folder.type_seq == {'train': None, 'valid': None, 'test': None}
        assert dataset_no_dataset_folder.type_to_idx == \
            {'word': None, 'char': None, 'ent': None, 'rel': None}
        assert dataset_no_dataset_folder.idx_to_tag == {'ent': None, 'rel': None}
        assert dataset_no_dataset_folder.idx_seq == {'train': None, 'valid': None, 'test': None}

    def test_attributes_after_init(self, dataset):
        """Asserts that the attributes of a `Dataset` object are initialized with the expected
        values when a dataset folder is provided.
        """
        # Attributes that are passed to __init__
        for partition in dataset.dataset_folder:
            expected = os.path.join(PATH_TO_CONLL2003_DATASET, '{}.tsv'.format(partition))
            assert dataset.dataset_folder[partition] == expected
        assert not dataset.replace_rare_tokens

        # Other instance attributes
        assert dataset.type_seq == {'train': None, 'valid': None, 'test': None}
        assert dataset.type_to_idx == \
            {'word': None, 'char': None, 'ent': None, 'rel': None}
        assert dataset.idx_to_tag == {'ent': None, 'rel': None}
        assert dataset.idx_seq == {'train': None, 'valid': None, 'test': None}

    def test_value_error_load(self, dataset_no_dataset_folder):
        """Asserts that `Dataset.load()` raises a ValueError when `Dataset.dataset_folder` is None.
        """
        with pytest.raises(ValueError):
            dataset_no_dataset_folder.load()


class TestCoNLL2003DatasetReader(object):
    """Collects all unit tests for `saber.dataset.CoNLL2003DatasetReader`.
    """
    def test_attributes_after_init_no_dataset_folder(self,
                                                     conll2003datasetreader_no_dataset_folder):
        """Asserts that the attributes of a `Dataset` object are initialized with the expected
        values when no dataset folder is provided.
        """
        # Attributes that are passed to __init__
        assert conll2003datasetreader_no_dataset_folder.dataset_folder is None
        assert not conll2003datasetreader_no_dataset_folder.replace_rare_tokens

        # Other instance attributes
        assert conll2003datasetreader_no_dataset_folder.type_seq == \
            {'train': None, 'valid': None, 'test': None}
        assert conll2003datasetreader_no_dataset_folder.type_to_idx == \
            {'word': None, 'char': None, 'ent': None, 'rel': None}
        assert conll2003datasetreader_no_dataset_folder.idx_to_tag == {'ent': None, 'rel': None}
        assert conll2003datasetreader_no_dataset_folder.idx_seq == \
            {'train': None, 'valid': None, 'test': None}

    def test_attributes_after_init(self, conll2003datasetreader):
        """Asserts instance attributes are initialized correctly when dataset is empty (i.e.,
        `Dataset.load()` has not been called).
        """
        # Attributes that are passed to __init__
        for partition in conll2003datasetreader.dataset_folder:
            expected = os.path.join(PATH_TO_CONLL2003_DATASET, '{}.tsv'.format(partition))
            actual = conll2003datasetreader.dataset_folder[partition]
            assert expected == actual
        assert not conll2003datasetreader.replace_rare_tokens

        # Other instance attributes
        assert conll2003datasetreader.type_seq == \
            {'train': None, 'valid': None, 'test': None}
        assert conll2003datasetreader.type_to_idx == \
            {'word': None, 'char': None, 'ent': None, 'rel': None}
        assert conll2003datasetreader.idx_to_tag == {'ent': None, 'rel': None}
        assert conll2003datasetreader.idx_seq == \
            {'train': None, 'valid': None, 'test': None}

    def test_value_error_load(self, conll2003datasetreader_no_dataset_folder):
        """Asserts that `Dataset.load()` raises a ValueError when `Dataset.dataset_folder` is None.
        """
        with pytest.raises(ValueError):
            conll2003datasetreader_no_dataset_folder.load()

    def test_get_type_seq(self, conll2003datasetreader):
        """Asserts that `Dataset.type_seq` is updated as expected after call to
        `Dataset._get_type_seq()`.
        """
        conll2003datasetreader._get_type_seq()

        assert np.array_equal(conll2003datasetreader.type_seq['train']['word'], CoNLL2003_WORD_SEQ)
        assert np.array_equal(conll2003datasetreader.type_seq['train']['char'], CoNLL2003_CHAR_SEQ)
        assert np.array_equal(conll2003datasetreader.type_seq['train']['ent'], CoNLL2003_ENT_SEQ)

    def test_get_types(self, conll2003datasetreader):
        """Asserts that `Dataset._get_types()` returns the expected values.
        """
        conll2003datasetreader._get_type_seq()

        expected = {
            'word': CoNLL2003_WORD_TYPES,
            'char': CoNLL2003_CHAR_TYPES,
            'ent': CoNLL2003_ENT_TYPES
        }
        actual = conll2003datasetreader._get_types()

        # sorted allows us to assert that the two lists are identical
        assert all(sorted(actual[k]) == sorted(v) for k, v in expected.items())

    def test_get_idx_maps(self, conll2003datasetreader):
        """Asserts that `Dataset.type_to_idx` is updated as expected after successive calls to
        `Dataset._get_types()` and `Dataset._get_idx_maps()`.
        """
        conll2003datasetreader._get_type_seq()
        types = conll2003datasetreader._get_types()

        conll2003datasetreader._get_idx_maps(types)

        # Ensure that index mapping is a contigous sequence of numbers starting at 0
        assert generic_utils.is_consecutive(conll2003datasetreader.type_to_idx['word'].values())
        assert generic_utils.is_consecutive(conll2003datasetreader.type_to_idx['char'].values())
        assert generic_utils.is_consecutive(conll2003datasetreader.type_to_idx['ent'].values())
        # Ensure that type to index mapping contains the expected keys
        assert all(key in CoNLL2003_WORD_TYPES for key in
                   conll2003datasetreader.type_to_idx['word'])
        assert all(key in CoNLL2003_CHAR_TYPES for key in
                   conll2003datasetreader.type_to_idx['char'])
        assert all(key in CoNLL2003_ENT_TYPES for key in
                   conll2003datasetreader.type_to_idx['ent'])

        # Assert special tokens are mapped to the correct indices
        assert all(conll2003datasetreader.type_to_idx['word'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2003datasetreader.type_to_idx['char'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2003datasetreader.type_to_idx['ent'][k] == v
                   for k, v in constants.INITIAL_MAPPING['ent'].items())

    def test_get_idx_seq(self, conll2003datasetreader):
        """Asserts that `Dataset.idx_seq` is updated as expected after successive calls to
        `Dataset._get_type_seq()`, `Dataset._get_idx_maps()` and `Dataset.get_idx_seq()`.
        """
        conll2003datasetreader._get_type_seq()
        types = conll2003datasetreader._get_types()
        conll2003datasetreader._get_idx_maps(types)

        conll2003datasetreader.get_idx_seq()

        # As a workaround to testing this directly, just check that shapes are as expected
        expected_word_idx_shape = (len(CoNLL2003_WORD_SEQ), MAX_SENT_LEN)
        expected_char_idx_shape = (len(CoNLL2003_WORD_SEQ), MAX_SENT_LEN, MAX_CHAR_LEN)
        expected_ent_idx_shape = (len(CoNLL2003_WORD_SEQ), MAX_SENT_LEN)

        assert all(conll2003datasetreader.idx_seq[partition]['word'].shape ==
                   expected_word_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2003datasetreader.idx_seq[partition]['char'].shape ==
                   expected_char_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2003datasetreader.idx_seq[partition]['ent'].shape == expected_ent_idx_shape
                   for partition in ['train', 'test', 'valid'])

    def test_get_type_seq_after_load(self, conll2003datasetreader_load):
        """Asserts that `Dataset.type_seq` is updated as expected after call to `Dataset.load()`.
        """
        assert np.array_equal(conll2003datasetreader_load.type_seq['train']['word'],
                              CoNLL2003_WORD_SEQ)
        assert np.array_equal(conll2003datasetreader_load.type_seq['train']['char'],
                              CoNLL2003_CHAR_SEQ)
        assert np.array_equal(conll2003datasetreader_load.type_seq['train']['ent'],
                              CoNLL2003_ENT_SEQ)

    def test_get_idx_maps_after_load(self, conll2003datasetreader_load):
        """Asserts that `Dataset.type_to_idx` is updated as expected after call to `Dataset.load()`.
        """
        # Ensure that index mapping is a contigous sequence of numbers starting at 0
        assert generic_utils.is_consecutive(
            conll2003datasetreader_load.type_to_idx['word'].values()
        )
        assert generic_utils.is_consecutive(
            conll2003datasetreader_load.type_to_idx['char'].values()
        )
        assert generic_utils.is_consecutive(
            conll2003datasetreader_load.type_to_idx['ent'].values()
        )
        # Ensure that type to index mapping contains the expected keys
        assert all(key in CoNLL2003_WORD_TYPES for key in
                   conll2003datasetreader_load.type_to_idx['word'])
        assert all(key in CoNLL2003_CHAR_TYPES for key in
                   conll2003datasetreader_load.type_to_idx['char'])
        assert all(key in CoNLL2003_ENT_TYPES for key in
                   conll2003datasetreader_load.type_to_idx['ent'])

        # Assert special tokens are mapped to the correct indices
        assert all(conll2003datasetreader_load.type_to_idx['word'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2003datasetreader_load.type_to_idx['char'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2003datasetreader_load.type_to_idx['ent'][k] == v
                   for k, v in constants.INITIAL_MAPPING['ent'].items())

    def test_get_idx_seq_after_load(self, conll2003datasetreader_load):
        """Asserts that `Dataset.idx_seq` is updated as expected after calls to `Dataset.load()`.
        """
        # As a workaround to testing this directly, just check that shapes are as expected
        expected_word_idx_shape = (len(CoNLL2003_WORD_SEQ), MAX_SENT_LEN)
        expected_char_idx_shape = (len(CoNLL2003_WORD_SEQ), MAX_SENT_LEN, MAX_CHAR_LEN)
        expected_ent_idx_shape = (len(CoNLL2003_WORD_SEQ), MAX_SENT_LEN)

        assert all(conll2003datasetreader_load.idx_seq[partition]['word'].shape ==
                   expected_word_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2003datasetreader_load.idx_seq[partition]['char'].shape ==
                   expected_char_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2003datasetreader_load.idx_seq[partition]['ent'].shape ==
                   expected_ent_idx_shape for partition in ['train', 'test', 'valid'])


class TestCoNLL2004DatasetReader(object):
    """Collects all unit tests for `saber.dataset.CoNLL2003DatasetReader`.
    """
    def test_attributes_after_init_no_dataset_folder(self,
                                                     conll2004datasetreader_no_dataset_folder):
        """Asserts that the attributes of a `Dataset` object are initialized with the expected
        values when no dataset folder is provided.
        """
        # Attributes that are passed to __init__
        assert conll2004datasetreader_no_dataset_folder.dataset_folder is None
        assert not conll2004datasetreader_no_dataset_folder.replace_rare_tokens

        # Other instance attributes
        assert conll2004datasetreader_no_dataset_folder.type_seq == \
            {'train': None, 'valid': None, 'test': None}
        assert conll2004datasetreader_no_dataset_folder.type_to_idx == \
            {'word': None, 'char': None, 'ent': None, 'rel': None}
        assert conll2004datasetreader_no_dataset_folder.idx_to_tag == {'ent': None, 'rel': None}
        assert conll2004datasetreader_no_dataset_folder.idx_seq == \
            {'train': None, 'valid': None, 'test': None}

    def test_attributes_after_init(self, conll2004datasetreader):
        """Asserts instance attributes are initialized correctly when dataset is empty (i.e.,
        `Dataset.load()` has not been called).
        """
        # Attributes that are passed to __init__
        for partition in conll2004datasetreader.dataset_folder:
            expected = os.path.join(PATH_TO_CONLL2004_DATASET, '{}.tsv'.format(partition))
            actual = conll2004datasetreader.dataset_folder[partition]
            assert expected == actual
        assert not conll2004datasetreader.replace_rare_tokens

        # Other instance attributes
        assert conll2004datasetreader.type_seq == \
            {'train': None, 'valid': None, 'test': None}
        assert conll2004datasetreader.type_to_idx == \
            {'word': None, 'char': None, 'ent': None, 'rel': None}
        assert conll2004datasetreader.idx_to_tag == {'ent': None, 'rel': None}
        assert conll2004datasetreader.idx_seq == \
            {'train': None, 'valid': None, 'test': None}

    def test_value_error_load(self, conll2004datasetreader_no_dataset_folder):
        """Asserts that `Dataset.load()` raises a ValueError when `Dataset.dataset_folder` is None.
        """
        with pytest.raises(ValueError):
            conll2004datasetreader_no_dataset_folder.load()

    def test_get_type_seq(self, conll2004datasetreader):
        """Asserts that `Dataset.type_seq` is updated as expected after call to
        `Dataset._get_type_seq()`.
        """
        conll2004datasetreader._get_type_seq()

        assert np.array_equal(conll2004datasetreader.type_seq['train']['word'], CoNLL2004_WORD_SEQ)
        assert np.array_equal(conll2004datasetreader.type_seq['train']['char'], CoNLL2004_CHAR_SEQ)
        assert np.array_equal(conll2004datasetreader.type_seq['train']['ent'], CoNLL2004_ENT_SEQ)
        assert conll2004datasetreader.type_seq['train']['rel'] == CoNLL2004_REL_SEQ

    def test_get_types(self, conll2004datasetreader):
        """Asserts that `Dataset._get_types()` returns the expected values.
        """
        conll2004datasetreader._get_type_seq()

        expected = {'word': CoNLL2004_WORD_TYPES,
                    'char': CoNLL2004_CHAR_TYPES,
                    'ent': CoNLL2004_ENT_TYPES,
                    'rel': CoNLL2004_REL_TYPES}
        actual = conll2004datasetreader._get_types()

        # sorted allows us to assert that the two lists are identical
        assert all(sorted(actual[k]) == sorted(v) for k, v in expected.items())

    def test_get_idx_maps(self, conll2004datasetreader):
        """Asserts that `Dataset.type_to_idx` is updated as expected after successive calls to
        `Dataset._get_types()` and `Dataset._get_idx_maps()`.
        """
        conll2004datasetreader._get_type_seq()
        types = conll2004datasetreader._get_types()

        conll2004datasetreader._get_idx_maps(types)

        # Ensure that index mapping is a contigous sequence of numbers starting at 0
        assert generic_utils.is_consecutive(conll2004datasetreader.type_to_idx['word'].values())
        assert generic_utils.is_consecutive(conll2004datasetreader.type_to_idx['char'].values())
        assert generic_utils.is_consecutive(conll2004datasetreader.type_to_idx['ent'].values())
        assert generic_utils.is_consecutive(conll2004datasetreader.type_to_idx['rel'].values())
        # Ensure that type to index mapping contains the expected keys
        assert all(key in CoNLL2004_WORD_TYPES for key in
                   conll2004datasetreader.type_to_idx['word'])
        assert all(key in CoNLL2004_CHAR_TYPES for key in
                   conll2004datasetreader.type_to_idx['char'])
        assert all(key in CoNLL2004_ENT_TYPES for key in
                   conll2004datasetreader.type_to_idx['ent'])
        assert all(key in CoNLL2004_REL_TYPES for key in
                   conll2004datasetreader.type_to_idx['rel'])

        # Assert special tokens are mapped to the correct indices
        assert all(conll2004datasetreader.type_to_idx['word'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2004datasetreader.type_to_idx['char'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2004datasetreader.type_to_idx['ent'][k] == v
                   for k, v in constants.INITIAL_MAPPING['ent'].items())
        assert all(conll2004datasetreader.type_to_idx['rel'][k] == v
                   for k, v in constants.INITIAL_MAPPING['rel'].items())

    def test_get_idx_seq(self, conll2004datasetreader):
        """Asserts that `Dataset.idx_seq` is updated as expected after successive calls to
        `Dataset._get_type_seq()`, `Dataset._get_idx_maps()` and `Dataset.get_idx_seq()`.
        """
        conll2004datasetreader._get_type_seq()
        types = conll2004datasetreader._get_types()
        conll2004datasetreader._get_idx_maps(types)

        conll2004datasetreader.get_idx_seq()

        # As a workaround to testing this directly, just check that shapes are as expected
        expected_word_idx_shape = (len(CoNLL2004_WORD_SEQ), MAX_SENT_LEN)
        expected_char_idx_shape = (len(CoNLL2004_WORD_SEQ), MAX_SENT_LEN, MAX_CHAR_LEN)
        expected_ent_idx_shape = (len(CoNLL2004_WORD_SEQ), MAX_SENT_LEN)

        assert all(conll2004datasetreader.idx_seq[partition]['word'].shape ==
                   expected_word_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2004datasetreader.idx_seq[partition]['char'].shape ==
                   expected_char_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2004datasetreader.idx_seq[partition]['ent'].shape == expected_ent_idx_shape
                   for partition in ['train', 'test', 'valid'])

        # TODO (John): This works for now but will break if the dummy dataset had multiple
        # relation classes.
        expected_rel_idx = [[[0, 11, 1], [3, 11, 1]], [], []]
        assert all(conll2004datasetreader.idx_seq[partition]['rel'] == expected_rel_idx
                   for partition in ['train', 'test', 'valid'])

    def test_get_type_seq_after_load(self, conll2004datasetreader_load):
        """Asserts that `Dataset.type_seq` is updated as expected after call to `Dataset.load()`.
        """
        assert np.array_equal(conll2004datasetreader_load.type_seq['train']['word'],
                              CoNLL2004_WORD_SEQ)
        assert np.array_equal(conll2004datasetreader_load.type_seq['train']['char'],
                              CoNLL2004_CHAR_SEQ)
        assert np.array_equal(conll2004datasetreader_load.type_seq['train']['ent'],
                              CoNLL2004_ENT_SEQ)
        assert conll2004datasetreader_load.type_seq['train']['rel'] == CoNLL2004_REL_SEQ

    def test_get_idx_maps_after_load(self, conll2004datasetreader_load):
        """Asserts that `Dataset.type_to_idx` is updated as expected after call to `Dataset.load()`.
        """
        # Ensure that index mapping is a contigous sequence of numbers starting at 0
        assert generic_utils.is_consecutive(
            conll2004datasetreader_load.type_to_idx['word'].values()
        )
        assert generic_utils.is_consecutive(
            conll2004datasetreader_load.type_to_idx['char'].values()
        )
        assert generic_utils.is_consecutive(conll2004datasetreader_load.type_to_idx['ent'].values())
        assert generic_utils.is_consecutive(conll2004datasetreader_load.type_to_idx['rel'].values())
        # Ensure that type to index mapping contains the expected keys
        assert all(key in CoNLL2004_WORD_TYPES for key in
                   conll2004datasetreader_load.type_to_idx['word'])
        assert all(key in CoNLL2004_CHAR_TYPES for key in
                   conll2004datasetreader_load.type_to_idx['char'])
        assert all(key in CoNLL2004_ENT_TYPES for key in
                   conll2004datasetreader_load.type_to_idx['ent'])
        assert all(key in CoNLL2004_REL_TYPES for key in
                   conll2004datasetreader_load.type_to_idx['rel'])

        # Assert special tokens are mapped to the correct indices
        assert all(conll2004datasetreader_load.type_to_idx['word'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2004datasetreader_load.type_to_idx['char'][k] == v
                   for k, v in constants.INITIAL_MAPPING['word'].items())
        assert all(conll2004datasetreader_load.type_to_idx['ent'][k] == v
                   for k, v in constants.INITIAL_MAPPING['ent'].items())
        assert all(conll2004datasetreader_load.type_to_idx['rel'][k] == v
                   for k, v in constants.INITIAL_MAPPING['rel'].items())

    def test_get_idx_seq_after_load(self, conll2004datasetreader_load):
        """Asserts that `Dataset.idx_seq` is updated as expected after calls to `Dataset.load()`.
        """
        # As a workaround to testing this directly, just check that shapes are as expected
        expected_word_idx_shape = (len(CoNLL2004_WORD_SEQ), MAX_SENT_LEN)
        expected_char_idx_shape = (len(CoNLL2004_WORD_SEQ), MAX_SENT_LEN,
                                   constants.MAX_CHAR_LEN)
        expected_ent_idx_shape = (len(CoNLL2004_WORD_SEQ), MAX_SENT_LEN)

        assert all(conll2004datasetreader_load.idx_seq[partition]['word'].shape ==
                   expected_word_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2004datasetreader_load.idx_seq[partition]['char'].shape ==
                   expected_char_idx_shape for partition in ['train', 'test', 'valid'])
        assert all(conll2004datasetreader_load.idx_seq[partition]['ent'].shape ==
                   expected_ent_idx_shape for partition in ['train', 'test', 'valid'])

        # TODO (John): This works for now but will break if the dummy dataset had multiple
        # relation classes.
        expected_rel_idx = [[[0, 11, 1], [3, 11, 1]], [], []]
        assert all(conll2004datasetreader_load.idx_seq[partition]['rel'] == expected_rel_idx
                   for partition in ['train', 'test', 'valid'])
