"""Any and all unit tests for the MultiTaskLSTMCRF (saber/models/multi_task_lstm_crf.py).
"""
from keras.engine.training import Model

import pytest

from ..config import Config
from ..dataset import Dataset
from ..embeddings import Embeddings
from ..models.base_model import BaseKerasModel
from ..models.multi_task_lstm_crf import MultiTaskLSTMCRF
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    return Config(PATH_TO_DUMMY_CONFIG)

@pytest.fixture
def dummy_dataset_1():
    """Returns a single dummy Dataset instance after calling `Dataset.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False)
    dataset.load()

    return dataset

@pytest.fixture
def dummy_dataset_2():
    """Returns a single dummy Dataset instance after calling `Dataset.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET_2, replace_rare_tokens=False)
    dataset.load()

    return dataset

@pytest.fixture
def dummy_embeddings(dummy_dataset_1):
    """Returns an instance of an `Embeddings()` object AFTER the `.load()` method is called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                            token_map=dummy_dataset_1.idx_to_tag)
    embeddings.load(binary=False) # txt file format is easier to test
    return embeddings

@pytest.fixture
def single_model(dummy_config, dummy_dataset_1, dummy_embeddings):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration."""
    model = MultiTaskLSTMCRF(config=dummy_config,
                             datasets=[dummy_dataset_1],
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')
    return model

@pytest.fixture
def single_model_specify(single_model):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file and
    a single specified model."""
    single_model.specify()

    return single_model

@pytest.fixture
def single_model_embeddings(dummy_config, dummy_dataset_1, dummy_embeddings):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file and
    loaded embeddings"""
    model = MultiTaskLSTMCRF(config=dummy_config,
                             datasets=[dummy_dataset_1],
                             embeddings=dummy_embeddings,
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')
    return model

@pytest.fixture
def single_model_embeddings_specify(single_model_embeddings):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file,
    loaded embeddings and single specified model."""
    single_model_embeddings.specify()

    return single_model_embeddings

############################################ UNIT TESTS ############################################

def test_attributes_init_of_single_model(dummy_config, dummy_dataset_1, single_model):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(single_model, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_model.config is dummy_config
    assert single_model.datasets[0] is dummy_dataset_1
    assert single_model.embeddings is None
    # other instance attributes
    assert single_model.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_model.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_model_specify(dummy_config, dummy_dataset_1, single_model_specify):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF`
    model is initialized without embeddings (`embeddings` attribute is None) and
    `MultiTaskLSTMCRF.specify()` has been called.
    """
    assert isinstance(single_model_specify, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_model_specify.config is dummy_config
    assert single_model_specify.datasets[0] is dummy_dataset_1
    assert single_model_specify.embeddings is None
    # other instance attributes
    assert all([isinstance(model, Model) for model in single_model_specify.models])
    # test that we can pass arbitrary keyword arguments
    assert single_model_specify.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_model_embeddings(dummy_config, dummy_dataset_1,
                                                    dummy_embeddings, single_model_embeddings):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized with embeddings (`embeddings` attribute is not None.)
    """
    assert isinstance(single_model_embeddings, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_model_embeddings.config is dummy_config
    assert single_model_embeddings.datasets[0] is dummy_dataset_1
    assert single_model_embeddings.embeddings is dummy_embeddings
    # other instance attributes
    assert single_model_embeddings.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_model_embeddings.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_model_embeddings_specify(dummy_config,
                                                            dummy_dataset_1,
                                                            dummy_embeddings,
                                                            single_model_embeddings_specify):
    """Asserts instance attributes are initialized correctly when single MultiTaskLSTMCRF
    model is initialized with embeddings (`embeddings` attribute is not None) and
    `MultiTaskLSTMCRF.specify()` has been called.
    """
    assert isinstance(single_model_embeddings_specify, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_model_embeddings_specify.config is dummy_config
    assert single_model_embeddings_specify.datasets[0] is dummy_dataset_1
    assert single_model_embeddings_specify.embeddings is dummy_embeddings
    # other instance attributes
    assert all([isinstance(model, Model) for model in single_model_embeddings_specify.models])
    # test that we can pass arbitrary keyword arguments
    assert single_model_embeddings_specify.totally_arbitrary == 'arbitrary'

def test_crf_after_transfer(single_model_specify, dummy_dataset_2):
    """Asserts that the CRF output layer of a model is replaced with a new layer when
    `MultiTaskLSTMCRF.prepare_for_transfer()` is called by testing that the `name` attribute
    of the final layer.
    """
    # shorten test statements
    test_model = single_model_specify

    # get output layer names before transfer
    expected_before_transfer = ['crf_classifier']
    actual_before_transfer = [model.layers[-1].name for model in test_model.models]
    # get output layer names after transfer
    test_model.prepare_for_transfer([dummy_dataset_2])
    expected_after_transfer = ['target_crf_classifier']
    actual_after_transfer = [model.layers[-1].name for model in test_model.models]

    assert actual_before_transfer == expected_before_transfer
    assert actual_after_transfer == expected_after_transfer
