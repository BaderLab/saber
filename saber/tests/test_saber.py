"""Test suite for the `Saber` class (saber.saber.Saber).
"""
import os
from glob import glob

import pytest

from .. import constants
from ..dataset import Dataset
from ..models.base_model import BaseModel
from ..models.bert_for_ner import BertForNER
from ..saber import MissingStepException
from .resources.constants import PATH_TO_CONLL2003_DATASET
from .resources.constants import PATH_TO_DUMMY_DATASET_2

# TODO (johngiorgi): Write tests for compound dataset


class TestSaber(object):
    """Collects all unit tests for `saber.saber.Saber`
    """
    def test_attributes_after_initilization_of_model(self, saber_blank, dummy_config):
        """Asserts instance attributes are initialized correctly when `Saber` object is created.
        """
        assert saber_blank.config is dummy_config

        assert saber_blank.preprocessor is None
        assert saber_blank.datasets == []
        assert saber_blank.models == []

    def test_annotate_value_error(self, saber_bert_for_ner):
        """Asserts that call to `Saber.annotate()` raises a ValueError when an empty/falsey value for
        `text` is passed.
        loaded."""
        tests = ['', None, 0, []]

        for test in tests:
            with pytest.raises(ValueError):
                saber_bert_for_ner.annotate(text=test)

    def test_annotate_with_bert_for_ner(self, saber_bert_for_ner):
        """Asserts that call to `Saber.annotate()` returns the expected results with a single dataset
        loaded."""
        test = "This is a simple test. With multiple sentences."
        expected = {'text': test, 'title': '', 'ents': [],
                    # TODO (John): This is temp, fixes SpaCy bug.
                    'settings': {}}

        actual = saber_bert_for_ner.annotate(test)
        actual['ents'] = []  # wipe the predicted ents as they are stochastic.

        assert expected == actual

    def test_annotate_with_bert_for_ner_mt(self, saber_bert_for_ner_mt):
        """Asserts that call to `Saber.annotate()` returns the expected results with a BertForNER
        model loaded."""
        test = "This is a simple test. With multiple sentences."
        expected = {'text': test, 'title': '', 'ents': [],
                    # TODO (John): This is temp, fixes SpaCy bug.
                    'settings': {}}

        actual = saber_bert_for_ner_mt.annotate(test)
        actual['ents'] = []  # wipe the predicted ents as they are stochastic.

        assert expected == actual

    # TODO (John): Implement this when we settle on an interface for annotation
    def test_annotate_with_bert_for_ner_and_re(self, saber_bert_for_ner_and_re):
        """Asserts that call to `Saber.annotate()` returns the expected results with a
        BertForNERAndRE model loaded."""
        pass

    # TODO (John): Implement this when we get the MT model working
    def test_annotate_with_bert_for_ner_and_re_mt(self, saber_bert_for_ner_and_re):
        """Asserts that call to `Saber.annotate()` returns the expected results with a multi-task
        BertForNERAndRE model loaded."""
        pass

    def test_save_missing_step_exception(self, saber_blank):
        """Asserts that `Saber` object raises a MissingStepException when we try to call `Saber.save()`
        without first loading a model (`Saber.models` is []).
        """
        with pytest.raises(MissingStepException):
            saber_blank.save()

    def test_save_file_exists(self, tmpdir, saber_bert_for_ner):
        """Asserts that a file is created under the expected directory when `Saber.save()` is called.
        """
        model_save_dir = tmpdir.mkdir("saved_model")
        saved_directory = saber_bert_for_ner.save(directory=model_save_dir, compress=True)

        assert os.path.exists(f'{saved_directory}.tar.bz2')

    def test_save_dir_exists(self, tmpdir, saber_bert_for_ner):
        """Asserts that a directory is created under the expected directory when
        `Saber.save(compress=False)` is called.
        """
        model_save_dir = tmpdir.mkdir("saved_model")
        saved_directory = saber_bert_for_ner.save(directory=model_save_dir, compress=False)

        model_attributes_filepath = os.path.join(saved_directory, constants.ATTRIBUTES_FILENAME)
        model_filepath = \
            glob(os.path.join(saved_directory, f'*{constants.PRETRAINED_MODEL_FILENAME}'))[0]

        assert os.path.isdir(saved_directory)
        assert os.path.isfile(model_attributes_filepath)
        assert os.path.isfile(model_filepath)

    def test_load(self, saber_saved_model):
        """Tests that the attributes of a loaded model are as expected after `Saber.load()` is called.
        """
        saber, model, dataset, directory = saber_saved_model
        saber.load(directory)

        assert saber.models[-1].config == model.config
        assert saber.models[-1].datasets[-1].type_to_idx == dataset.type_to_idx
        assert saber.models[-1].datasets[-1].idx_to_tag == dataset.idx_to_tag

    def test_load_single_dataset(self, saber_single_dataset):
        """Assert that the `datasets` attribute of a `Saber` instance was updated as expected after
        call to `Saber.load_dataset()` when a single dataset was provided.
        """
        assert all([isinstance(ds, Dataset) for ds in saber_single_dataset.datasets])

    def test_load_compound_dataset(self, saber_compound_dataset):
        """Assert that the `datasets` attribute of a `Saber` instance was updated as expected after
        call to `Saber.load_dataset()` when a compound dataset was provided.
        """
        assert all([isinstance(ds, Dataset) for ds in saber_compound_dataset.datasets])

    def test_load_dataset_value_error(self, saber_single_dataset):
        """Asserts that `Saber` object raises a ValueError when we try to load a dataset but have
        not specified a path to that dataset (`Saber.config.dataset_folder` is False).
        """
        # set dataset_folder argument to empty string so we can test exception
        saber_single_dataset.config.dataset_folder = ''
        with pytest.raises(ValueError):
            saber_single_dataset.load_dataset()

    def test_tag_to_idx_after_load_single_dataset_with_transfer(self,
                                                                dummy_dataset_2,
                                                                saber_bert_for_ner):
        """Asserts that `saber.datasets[0].type_to_idx['ent']` is unchanged after we load a single
        target dataset for transfer learning.
        """
        '''
        expected = dummy_dataset_2.type_to_idx['ent']
        saber_bert_for_ner.load_dataset(PATH_TO_DUMMY_DATASET_2)
        actual = saber_bert_for_ner.datasets[0].type_to_idx['ent']

        assert actual == expected
        '''
        with pytest.raises(NotImplementedError):
            saber_bert_for_ner.load_dataset(PATH_TO_DUMMY_DATASET_2)

    def test_tag_to_idx_after_load_compound_dataset_with_transfer(self,
                                                                  conll2003datasetreader_load,
                                                                  dummy_dataset_2,
                                                                  saber_bert_for_ner):
        """Asserts that `type_to_idx['ent']` is unchanged after we load a compound target dataset for
        transfer learning.
        """
        '''
        expected = [conll2003datasetreader_load.type_to_idx['ent'],
                    dummy_dataset_2.type_to_idx['ent']]
        saber_compound_dataset_model = saber_bert_for_ner
        saber_compound_dataset_model.load_dataset(
            [PATH_TO_CONLL2003_DATASET, PATH_TO_DUMMY_DATASET_2]
        )
        actual = [ds.type_to_idx['ent'] for ds in saber_compound_dataset_model.datasets]

        for i, result in enumerate(actual):
            assert result == expected[i]
        '''
        with pytest.raises(NotImplementedError):
            saber_compound_dataset_model = saber_bert_for_ner
            saber_compound_dataset_model.load_dataset(
                [PATH_TO_CONLL2003_DATASET, PATH_TO_DUMMY_DATASET_2]
            )

    def test_build_single_dataset(self, saber_single_dataset):
        """Assert that the `model` attribute of a `Saber` instance was updated as expected after
        call to `Saber.build()` when single dataset was loaded.
        """
        assert all(isinstance(model, (BaseModel, BertForNER)) for model in
                   saber_single_dataset.models)

    def test_build_compound_dataset(self, saber_compound_dataset):
        """Assert that the `model` attribute of a `Saber` instance was updated as expected after
        call to `Saber.build()` when compound dataset was loaded.
        """
        assert all(isinstance(model, (BaseModel, BertForNER)) for model in
                   saber_compound_dataset.models)

    def test_build_missing_step_exception(self, saber_blank):
        """Asserts that `Saber` object raises a MissingStepException when we try to build the model
        without first loading a dataset (`Saber.datasets` is None).
        """
        with pytest.raises(MissingStepException):
            saber_blank.build()

    def test_build_value_error(self, saber_single_dataset):
        """Asserts that `Saber` object raises a ValueError when we try to load a model with an invalid
        name (i.e. `Saber.config.model_name` is not in `constants.MODEL_NAMES`).
        """
        model_name = 'this is not valid'
        with pytest.raises(ValueError):
            saber_single_dataset.build(model_name)

    def test_train_no_model_missing_step_exception(self, saber_blank, saber_single_dataset):
        """Asserts that `Saber` object raises a MissingStepException when we try to train the model
        without first building the model (`Saber.models` is None).
        """
        with pytest.raises(MissingStepException):
            saber_blank.train()

        with pytest.raises(MissingStepException):
            saber_single_dataset.train()
