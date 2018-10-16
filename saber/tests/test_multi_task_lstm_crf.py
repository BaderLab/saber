import pytest

from ..config import Config
from ..dataset import Dataset
from ..embeddings import Embeddings
from ..models.multi_task_lstm_crf import MultiTaskLSTMCRF
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    return Config(PATH_TO_DUMMY_CONFIG)

@pytest.fixture
def dummy_dataset():
    """Returns a single dummy Dataset instance after calling Dataset.load().
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET, replace_rare=False)
    dataset.load()

    return dataset

@pytest.fixture
def dummy_embeddings(dummy_dataset):
    """Returns an instance of an Embeddings() object AFTER the `.load()` method is called.
    """
    embeddings = Embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS,
                            token_map=dummy_dataset.idx_to_tag)
    embeddings.load(binary=False) # txt file format is easier to test
    return embeddings

@pytest.fixture
def single_model_without_embeddings(dummy_config, dummy_dataset, dummy_embeddings):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file and
    a single compiled model."""
    model = MultiTaskLSTMCRF(config=dummy_config,
                             datasets=dummy_dataset,
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')

    return model

@pytest.fixture
def single_model_with_embeddings(dummy_config, dummy_dataset, dummy_embeddings):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file and
    a single compiled model."""
    model = MultiTaskLSTMCRF(config=dummy_config,
                             datasets=dummy_dataset,
                             embeddings=dummy_embeddings,
                             # to test passing of arbitrary keyword args to constructor
                             totally_arbitrary='arbitrary')

    return model

############################################ UNIT TESTS ############################################

def test_attributes_after_init_of_single_model_without_embeddings(dummy_config,
                                                                  dummy_dataset,
                                                                  single_model_without_embeddings):
    """Asserts instance attributes are initialized correctly when single MultiTaskLSTMCRF
    model is initialized without embeddings (`embeddings` attribute is not None.)
    """
    # attributes that are passed to __init__
    assert single_model_without_embeddings.config is dummy_config
    assert single_model_without_embeddings.datasets is dummy_dataset
    assert single_model_without_embeddings.embeddings is None
    # other instance attributes
    assert single_model_without_embeddings.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_model_without_embeddings.totally_arbitrary == 'arbitrary'

def test_attributes_after_init_of_single_model_with_embeddings(dummy_config,
                                                               dummy_dataset,
                                                               dummy_embeddings,
                                                               single_model_with_embeddings):
    """Asserts instance attributes are initialized correctly when single MultiTaskLSTMCRF
    model is initialized with embeddings (`embeddings` attribute is not None.)
    """
    # attributes that are passed to __init__
    assert single_model_with_embeddings.config is dummy_config
    assert single_model_with_embeddings.datasets is dummy_dataset
    assert single_model_with_embeddings.embeddings is dummy_embeddings
    # other instance attributes
    assert single_model_with_embeddings.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_model_with_embeddings.totally_arbitrary == 'arbitrary'
