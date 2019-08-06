"""Test suite for the `data_utils` module (saber.utils.data_utils).
"""
import pytest

from ..dataset import Dataset
from ..utils import data_utils


class TestDataUtils(object):
    """Collects all unit tests for `saber.utils.data_utils`.
    """
    def test_get_filepaths_value_error(self, tmpdir):
        """Asserts that a ValueError is raised when `data_utils.get_filepaths(tmpdir)` is called and
        no file '<tmpdir>/train.*' exists.
        """
        with pytest.raises(ValueError):
            data_utils.get_filepaths(tmpdir.strpath)

    def test_get_filepaths_all(self, dummy_dataset_paths):
        """Asserts that `data_utils.get_filepaths()` returns the expected filepaths when all partitions
        (train/test/valid) are provided.
        """
        dummy_dataset_folder, train_filepath, valid_filepath, test_filepath = dummy_dataset_paths
        expected = {'train': train_filepath,
                    'valid': valid_filepath,
                    'test': test_filepath
                    }
        actual = data_utils.get_filepaths(dummy_dataset_folder)

        assert actual == expected

    def test_get_filepaths_no_valid(self, dummy_dataset_paths_no_valid):
        """Asserts that `data_utils.get_filepaths()` returns the expected filepaths when train and
        test partitions are provided.
        """
        dummy_dataset_folder, train_filepath, test_filepath = dummy_dataset_paths_no_valid
        expected = {'train': train_filepath,
                    'valid': None,
                    'test': test_filepath
                    }
        actual = data_utils.get_filepaths(dummy_dataset_folder)

        assert actual == expected

    def test_load_single_dataset(self, dummy_config, conll2003datasetreader_load):
        """Asserts that `data_utils.load_single_dataset()` returns the expected value.
        """
        actual = data_utils.load_single_dataset(dummy_config)
        expected = [conll2003datasetreader_load]

        # essentially redundant, but if we dont return a [Dataset] object then the error message
        # from the final test could be cryptic
        assert isinstance(actual, list)
        assert len(actual) == 1
        assert isinstance(actual[0], Dataset)
        # the test we actually care about, least roundabout way of asking if the two Dataset objects
        # are identical
        assert dir(actual[0].__dict__) == dir(expected[0].__dict__)

    def test_load_compound_dataset_unchanged_attributes(self,
                                                        conll2003datasetreader_load,
                                                        dummy_dataset_2,
                                                        dummy_compound_dataset):
        """Asserts that attributes of `Dataset` objects which are expected to remain unchanged
        are unchanged after call to `data_utils.load_compound_dataset()`.
        """
        actual = dummy_compound_dataset
        expected = [conll2003datasetreader_load, dummy_dataset_2]

        # essentially redundant, but if we dont return a [Dataset, Dataset] object then the error
        # messages from the downstream tests could be cryptic
        assert isinstance(actual, list)
        assert len(actual) == 2
        assert all([isinstance(ds, Dataset) for ds in actual])

        # attributes that are unchanged in case of compound dataset
        assert actual[0].dataset_folder == expected[0].dataset_folder
        assert actual[0].replace_rare_tokens == expected[0].replace_rare_tokens
        assert actual[0].type_seq == expected[0].type_seq
        assert actual[0].type_to_idx['ent'] == expected[0].type_to_idx['ent']
        assert actual[0].idx_to_tag == expected[0].idx_to_tag

        assert actual[-1].dataset_folder == expected[-1].dataset_folder
        assert actual[-1].replace_rare_tokens == expected[-1].replace_rare_tokens
        assert actual[-1].type_seq == expected[-1].type_seq
        assert actual[-1].type_to_idx['ent'] == expected[-1].type_to_idx['ent']
        assert actual[-1].idx_to_tag == expected[-1].idx_to_tag

    def test_load_compound_dataset_changed_attributes(self,
                                                      conll2003datasetreader_load, dummy_dataset_2,
                                                      dummy_compound_dataset):
        """Asserts that attributes of `Dataset` objects which are expected to be changed are changed
        after call to `data_utils.load_compound_dataset()`.
        """
        actual = dummy_compound_dataset
        expected = [conll2003datasetreader_load, dummy_dataset_2]

        # essentially redundant, but if we don't return a [Dataset, Dataset] object then the error
        # messages from the downstream tests could be cryptic
        assert isinstance(actual, list)
        assert len(actual) == 2
        assert all([isinstance(ds, Dataset) for ds in actual])

        # attributes that are changed in case of compound dataset
        assert actual[0].type_to_idx['word'] == actual[-1].type_to_idx['word']
        assert actual[0].type_to_idx['char'] == actual[-1].type_to_idx['char']

        # TODO: Need to assert that all types in idx_seq map to the same integers
        # across the compound datasets

    def test_setup_dataset_for_transfer(self, conll2003datasetreader_load, dummy_dataset_2):
        """Asserts that the `type_to_idx` attribute of a "source" dataset and a "target" dataset are
        as expected after call to `data_utils.setup_dataset_for_transfer()`.
        """
        source_type_to_idx = conll2003datasetreader_load.type_to_idx
        data_utils.setup_dataset_for_transfer(dummy_dataset_2, source_type_to_idx)

        assert all(dummy_dataset_2.type_to_idx[type_] == source_type_to_idx[type_]
                   for type_ in ['word', 'char'])

    # TODO (John): This should check data directly instead of just looking at lens. Also,
    # we need additional tests for when validation_split > 0.
    def test_get_k_folds_no_validation_split(self, dummy_training_data):
        """Asserts that the correct number of training examples exist in each fold after a call
        to `data_utils.get_k_folds()`.
        """
        k_folds = 2
        actual = data_utils.get_k_folds(training_data=dummy_training_data, k_folds=k_folds)

        # Check that we created 2 folds
        assert len(actual) == k_folds

        # Check that the expected partitions exist with the correct number of training examples
        for fold in actual:
            # Train
            assert len(fold['train']['x'][0]) == 1
            assert len(fold['train']['x'][1]) == 1
            assert len(fold['train']['y']) == 1

            # Test
            assert len(fold['test']['x'][0]) == 1
            assert len(fold['test']['x'][1]) == 1
            assert len(fold['test']['y']) == 1

            # Valid
            assert not fold['valid']

    # TODO (John): This should check data directly instead of just looking at lens.
    def test_get_validation_split(self, dummy_training_data):
        """Asserts that the correct number of training examples exist in each fold after a call
        to `data_utils.get_k_folds()`.
        """
        actual = data_utils.get_validation_split(training_data=dummy_training_data,
                                                 validation_split=0.5)

        # Check that the expected partitions exist with the correct number of training examples
        # Train
        assert len(actual['train']['x'][0]) == 1
        assert len(actual['train']['x'][1]) == 1
        assert len(actual['train']['y']) == 1

        # Valid
        assert len(actual['valid']['x'][0]) == 1
        assert len(actual['valid']['x'][1]) == 1
        assert len(actual['valid']['y']) == 1

        # Test
        assert not actual['test']

    # TODO (John): This should check data directly instead of just looking at lens.
    # we need additional tests for when validation_split > 0.
    def test_prepare_data_for_eval_k_folds(self, dummy_config, dummy_training_data):
        """Asserts that the correct number of training examples exist in each fold after a call
        to `data_utils.get_k_folds()`.
        """
        dummy_config.k_folds = 2
        actual = data_utils.prepare_data_for_eval(config=dummy_config,
                                                  training_data=dummy_training_data)

        # Check that we created 2 folds
        assert len(actual) == dummy_config.k_folds

        # Check that the expected partitions exist with the correct number of training examples
        for fold in actual:
            # Train
            assert len(fold['train']['x'][0]) == 1
            assert len(fold['train']['x'][1]) == 1
            assert len(fold['train']['y']) == 1

            # Test
            assert len(fold['test']['x'][0]) == 1
            assert len(fold['test']['x'][1]) == 1
            assert len(fold['test']['y']) == 1

            # Valid
            assert not fold['valid']
