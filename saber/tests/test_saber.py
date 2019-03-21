"""Any and all unit tests for the `Saber` class (saber/utils/saber.py).
"""
import pytest

from ..dataset import Dataset
from ..embeddings import Embeddings
from ..models.base_model import BaseKerasModel
from ..preprocessor import Preprocessor
from ..saber import MissingStepException
from .resources import helpers
from .resources.dummy_constants import *

# TODO (johngiorgi): Write tests for compound dataset


def test_attributes_after_initilization_of_model(saber_blank,
                                                 dummy_config):
    """Asserts instance attributes are initialized correctly when `Saber` object is created.
    """
    assert saber_blank.config is dummy_config

    assert saber_blank.preprocessor is None

    assert saber_blank.datasets is None
    assert saber_blank.embeddings is None
    assert saber_blank.model is None

    # test that we can pass arbitrary keyword arguments
    assert saber_blank.totally_arbitrary == 'arbitrary'

# SINGLE DATASETS

def test_load_single_dataset(saber_single_dataset):
    """Assert that the `datasets` attribute of a `Saber` instance was updated as expected after
    call to `Saber.load_dataset()` when a single dataset was provided.
    """
    assert all([isinstance(ds, Dataset) for ds in saber_single_dataset.datasets])

def test_load_dataset_value_error(saber_single_dataset):
    """Asserts that `Saber` object raises a ValueError when we try to load a dataset but have
    not specified a path to that dataset (`Saber.config.dataset_folder` is False).
    """
    # set dataset_folder argument to empty string so we can test exception
    saber_single_dataset.config.dataset_folder = ''
    with pytest.raises(ValueError):
        saber_single_dataset.load_dataset()

def test_tag_to_idx_after_load_single_dataset_with_transfer(dummy_dataset_2,
                                                            saber_single_dataset_model):
    """Asserts that `saber.datasets[0].type_to_idx['tag']` is unchanged after we load a single
    target dataset for transfer learning.
    """
    expected = dummy_dataset_2.type_to_idx['tag']
    saber_single_dataset_model.load_dataset(PATH_TO_DUMMY_DATASET_2)
    actual = saber_single_dataset_model.datasets[0].type_to_idx['tag']

    assert actual == expected

def test_load_embeddings(saber_single_dataset_embeddings):
    """Assert that the `embeddings` attribute of a `Saber` instance was updated as expected after
    call to `Saber.load_embeddings()`
    """
    assert isinstance(saber_single_dataset_embeddings.embeddings, Embeddings)

def test_load_embeddings_with_load_all(saber_single_dataset):
    """Assert that the `datasets` and `embeddings` attributes of a `Saber` instance are updated as
    expected after call to `Saber.load_embeddings()`
    """
    # get the dataset object
    dataset = saber_single_dataset.datasets[0]
    # create our expected values
    word_types, char_types = list(dataset.type_to_idx['word']), list(dataset.type_to_idx['char'])
    expected = {'word': Preprocessor.type_to_idx(word_types, DUMMY_TOKEN_MAP),
                'char': Preprocessor.type_to_idx(char_types, DUMMY_CHAR_MAP)}

    # load the embeddings
    saber_single_dataset.load_embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                                         binary=False,
                                         load_all=True)

    # test for saber.datasets
    helpers.assert_type_to_idx_as_expected(dataset.type_to_idx, expected)
    # tests for saber.embeddings
    assert isinstance(saber_single_dataset.embeddings, Embeddings)


def test_load_embeddings_missing_step_exception(saber_blank):
    """Asserts that `Saber` object raises a MissingStepException when we try to load embeddings
    without first loading a dataset (`Saber.datasets` is None).
    """
    with pytest.raises(MissingStepException):
        saber_blank.load_embeddings()

def test_load_embeddings_value_error(saber_single_dataset):
    """Asserts that `Saber` object raises a ValueError when we try to load embeddings but have
    not specified a filepath to those embeddings (`Saber.config.pretrained_embeddings` is False).
    """
    # set pre-trained embedding argument to empty string so we can test exception
    saber_single_dataset.config.pretrained_embeddings = ''
    with pytest.raises(ValueError):
        saber_single_dataset.load_embeddings()

def test_build_single_dataset(saber_single_dataset_model):
    """Assert that the `model` attribute of a `Saber` instance was updated as expected after
    call to `Saber.build()` when single dataset was loaded.
    """
    assert isinstance(saber_single_dataset_model.model, BaseKerasModel)

def test_build_missing_step_exception(saber_blank):
    """Asserts that `Saber` object raises a MissingStepException when we try to build the model
    without first loading a dataset (`Saber.datasets` is None).
    """
    with pytest.raises(MissingStepException):
        saber_blank.build()

def test_build_value_error(saber_single_dataset):
    """Asserts that `Saber` object raises a ValueError when we try to load a model with an invalid
    name (i.e. `Saber.config.model_name` is not in `constants.MODEL_NAMES`).
    """
    model_name = 'this is not valid'
    with pytest.raises(ValueError):
        saber_single_dataset.build(model_name)

def test_train_no_dataset_missing_step_exception(saber_blank):
    """Asserts that `Saber` object raises a MissingStepException when we try to train the model
    without first loading a dataset (`Saber.datasets` is None).
    """
    with pytest.raises(MissingStepException):
        saber_blank.train()

def test_train_no_model_missing_step_exception(saber_single_dataset):
    """Asserts that `Saber` object raises a MissingStepException when we try to train the model
    without first building the model (`Saber.model` is None).
    """
    with pytest.raises(MissingStepException):
        saber_single_dataset.train()

def test_annotate_single(saber_single_dataset_model):
    """Asserts that call to `Saber.annotate()` returns the expected results with a single dataset
    loaded."""
    test = "This is a simple test. With multiple sentences"
    expected = {'text': test, 'title': '', 'ents': []}

    actual = saber_single_dataset_model.annotate(test)
    actual['ents'] = [] # wipe the predicted ents as they are stochastic.

    assert expected == actual

def test_predict_blank_or_invalid(saber_single_dataset_model):
    """Asserts that call to `Saber.predict()` raises a ValueError when a falsy text argument
    is passed."""
    # one test for each falsy type
    blank_text_test = ""
    none_test = None
    empty_list_test = []
    false_bool_test = False

    with pytest.raises(ValueError):
        saber_single_dataset_model.annotate(blank_text_test)
    with pytest.raises(ValueError):
        saber_single_dataset_model.annotate(none_test)
    with pytest.raises(ValueError):
        saber_single_dataset_model.annotate(empty_list_test)
    with pytest.raises(ValueError):
        saber_single_dataset_model.annotate(false_bool_test)

# COMPOUND DATASETS

def test_load_compound_dataset(saber_compound_dataset):
    """Assert that the `datasets` attribute of a `Saber` instance was updated as expected after
    call to `Saber.load_dataset()` when a compound dataset was provided.
    """
    assert all([isinstance(ds, Dataset) for ds in saber_compound_dataset.datasets])

def test_tag_to_idx_after_load_compound_dataset_with_transfer(dummy_dataset_1,
                                                              dummy_dataset_2,
                                                              saber_single_dataset_model):
    """Asserts that `type_to_idx['tag']` is unchanged after we load a compound target dataset for
    transfer learning.
    """
    expected = [dummy_dataset_1.type_to_idx['tag'],
                dummy_dataset_2.type_to_idx['tag']]
    saber_compound_dataset_model = saber_single_dataset_model
    saber_compound_dataset_model.load_dataset([PATH_TO_DUMMY_DATASET_1, PATH_TO_DUMMY_DATASET_2])
    actual = [ds.type_to_idx['tag'] for ds in saber_compound_dataset_model.datasets]

    for i, result in enumerate(actual):
        assert result == expected[i]

def test_build_compound_dataset(saber_compound_dataset_model):
    """Assert that the `model` attribute of a `Saber` instance was updated as expected after
    call to `Saber.build()` when compound dataset was loaded.
    """
    assert isinstance(saber_compound_dataset_model.model, BaseKerasModel)

'''
def test_annotate_compound(saber_compound_dataset_model):
    """Asserts that call to `Saber.annotate()` returns the expected results with a compound dataset
    loaded."""
    test = "This is a simple test. With multiple sentences"
    expected = {'text': test, 'ents': []}

    actual = saber_compound_dataset_model.annotate(test)
    actual['ents'] = [] # wipe the predicted ents as they are stochastic

    assert expected == actual
'''
