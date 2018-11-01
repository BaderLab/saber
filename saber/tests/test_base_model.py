"""Any and all unit tests for the BaseKerasModel (saber/models/base_model.py).
"""
import pytest

from ..config import Config
from ..dataset import Dataset
from ..embeddings import Embeddings
from ..models.base_model import BaseKerasModel
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
    model = BaseKerasModel(config=dummy_config,
                           datasets=[dummy_dataset_1],
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model

@pytest.fixture
def single_model_embeddings(dummy_config, dummy_dataset_1, dummy_embeddings):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file and
    loaded embeddings"""
    model = BaseKerasModel(config=dummy_config,
                           datasets=[dummy_dataset_1],
                           embeddings=dummy_embeddings,
                           # to test passing of arbitrary keyword args to constructor
                           totally_arbitrary='arbitrary')
    return model

############################################ UNIT TESTS ############################################

def test_compile_value_error(single_model):
    """Asserts that `BaseKerasModel._compile()` returns a ValueError when an invalid argument for
    `optimizer` is passed.
    """
    with pytest.raises(ValueError):
        single_model._compile('arbitrary', 'arbitrary', 'invalid')

def test_attributes_init_of_single_model(dummy_config, dummy_dataset_1, single_model):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(single_model, BaseKerasModel)
    # attributes that are passed to __init__
    assert single_model.config is dummy_config
    assert single_model.datasets[0] is dummy_dataset_1
    assert single_model.embeddings is None
    # other instance attributes
    assert single_model.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_model.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_model_embeddings(dummy_config, dummy_dataset_1,
                                                    dummy_embeddings, single_model_embeddings):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized with embeddings (`embeddings` attribute is not None.)
    """
    assert isinstance(single_model_embeddings, BaseKerasModel)
    # attributes that are passed to __init__
    assert single_model_embeddings.config is dummy_config
    assert single_model_embeddings.datasets[0] is dummy_dataset_1
    assert single_model_embeddings.embeddings is dummy_embeddings
    # other instance attributes
    assert single_model_embeddings.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_model_embeddings.totally_arbitrary == 'arbitrary'

def test_prepare_data_for_training(dummy_dataset_1, single_model):
    """Assert that the values returned from call to `BaseKerasModel.prepare_data_for_training()` are
    as expected.
    """
    training_data = single_model.prepare_data_for_training()
    partitions = ['x_train', 'y_train', 'x_valid', 'y_valid', 'x_test', 'y_test']

    # assert each item in training_data contains the expected keys
    assert all(partition in data for data in training_data for partition in partitions)

    # assert that the items in training_data contain the expected values
    assert all(data['x_train'] == [dummy_dataset_1.idx_seq['train']['word'], dummy_dataset_1.idx_seq['train']['char']]
               for data in training_data)
    assert all(data['x_valid'] == [dummy_dataset_1.idx_seq['valid']['word'], dummy_dataset_1.idx_seq['valid']['char']]
               for data in training_data)
    assert all(data['x_test'] == [dummy_dataset_1.idx_seq['test']['word'], dummy_dataset_1.idx_seq['test']['char']]
               for data in training_data)
    assert all(np.array_equal(data['y_train'], dummy_dataset_1.idx_seq['train']['tag']) for data in training_data)
    assert all(np.array_equal(data['y_valid'], dummy_dataset_1.idx_seq['valid']['tag']) for data in training_data)
    assert all(np.array_equal(data['y_test'], dummy_dataset_1.idx_seq['test']['tag']) for data in training_data)
