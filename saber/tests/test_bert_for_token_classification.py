"""Any and all unit tests for the BertForTokenClassification (saber/models/BertForTokenClassification.py).
"""
import pytest
from keras.engine.training import Model
from pytorch_pretrained_bert import BertForTokenClassification

from pytorch_pretrained_bert import BertTokenizer

from ..config import Config
from ..dataset import Dataset
from ..models.base_model import BaseModel, BasePyTorchModel
from ..models.bert_token_classifier import BertTokenClassifier
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture(scope='session')
def dummy_dir(tmpdir_factory):
    """Returns the path to a temporary directory.
    """
    dummy_dir = tmpdir_factory.mktemp('dummy_dir')
    return dummy_dir.strpath

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    return Config(PATH_TO_DUMMY_CONFIG)

@pytest.fixture
def dummy_dataset_1():
    """Returns a single dummy Dataset instance after calling `Dataset.load()`.
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(dataset_folder=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False)
    dataset.load()

    return dataset

@pytest.fixture
def single_model(dummy_config, dummy_dataset_1):
    """Returns an instance of BertForTokenClassification initialized with the default
    configuration."""
    model = BertTokenClassifier(config=dummy_config,
                                datasets=[dummy_dataset_1],
                                # to test passing of arbitrary keyword args to constructor
                                totally_arbitrary='arbitrary')
    return model

@pytest.fixture
def single_model_specify(single_model):
    """Returns an instance of BertForTokenClassification initialized with the default configuration
    file and a single specified model."""
    single_model.specify()

    return single_model

@pytest.fixture
def single_model_save(dummy_dir, single_model_specify):
    """Saves a model by calling `single_model_specify.save()` and returns the filepath to the saved
    model."""
    model_filepath = os.path.join(dummy_dir, constants.PYTORCH_MODEL_FILENAME)
    single_model_specify.save(model_filepath=model_filepath)

    return model_filepath

def test_attributes_init_of_single_model(dummy_config, dummy_dataset_1, single_model):
    """Asserts instance attributes are initialized correctly when single `BertTokenClassifier` model
    is initialized.
    """
    assert isinstance(single_model, (BertTokenClassifier, BasePyTorchModel, BaseModel))
    # attributes that are passed to __init__
    assert single_model.config is dummy_config
    assert single_model.datasets[0] is dummy_dataset_1
    # other instance attributes
    assert single_model.models == []
    assert single_model.embeddings is None
    assert single_model.device.type == 'cpu'
    assert single_model.pretrained_model_name_or_path == 'bert-base-uncased'
    assert isinstance(single_model.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_model.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_model_specify(dummy_config, dummy_dataset_1, single_model_specify):
    """Asserts instance attributes are initialized correctly when single `BertTokenClassifier`
    model is initialized without embeddings (`embeddings` attribute is None) and
    `BertTokenClassifier.specify()` has been called.
    """
    assert isinstance(single_model_specify, (BertTokenClassifier, BasePyTorchModel, BaseModel))
    # attributes that are passed to __init__
    assert single_model_specify.config is dummy_config
    assert single_model_specify.datasets[0] is dummy_dataset_1
    # other instance attributes
    assert all([isinstance(model, BertForTokenClassification) for model in single_model_specify.models])
    assert single_model_specify.embeddings is None
    assert single_model_specify.pretrained_model_name_or_path == 'bert-base-uncased'
    assert isinstance(single_model_specify.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_model_specify.totally_arbitrary == 'arbitrary'

def test_save(single_model_save):
    """Asserts that the expected file exists after call to `BertTokenClassifier.save()``.
    """
    model_filepath = single_model_save

    assert os.path.isfile(model_filepath)

def test_load(single_model, single_model_save, dummy_config, dummy_dataset_1):
    """Asserts that the attributes of a BertTokenClassifier object are expected after call to
    `BertTokenClassifier.load()`.
    """
    model_filepath = single_model_save

    single_model.load(model_filepath)

    assert isinstance(single_model, (BertTokenClassifier, BasePyTorchModel, BaseModel))
    # attributes that are passed to __init__
    assert single_model.config is dummy_config
    assert single_model.datasets[0] == dummy_dataset_1
    # other instance attributes
    assert all([isinstance(model, BertForTokenClassification) for model in single_model.models])
    assert single_model.embeddings is None
    assert single_model.pretrained_model_name_or_path == 'bert-base-uncased'
    assert isinstance(single_model.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_model.totally_arbitrary == 'arbitrary'
