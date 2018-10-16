"""Any and all unit tests for the `Saber` class (saber/utils/saber.py).
"""
import pytest

from ..config import Config
from ..dataset import Dataset
from ..embeddings import Embeddings
from ..models.base_model import BaseKerasModel
from ..saber import MissingStepException, Saber
from .resources.dummy_constants import *

# TODO (johngiorgi): Write tests for compound dataset

######################################### PYTEST FIXTURES #########################################

# SINGLE DATASET

@pytest.fixture
def dummy_config_single_dataset():
    """Returns instance of `Config` after parsing the dummy config file. Ensures that
    `replace_rare_tokens` argument is False.
    """
    cli_arguments = {'replace_rare_tokens': False}
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def saber_blank(dummy_config_single_dataset):
    """Returns instance of `Saber` initialized with the dummy config file and no dataset.
    """
    return Saber(config=dummy_config_single_dataset,
                 # to test passing of arbitrary keyword args to constructor
                 totally_arbitrary='arbitrary')

@pytest.fixture
def saber_single_dataset(dummy_config_single_dataset):
    """Returns instance of `Saber` initialized with the dummy config file and a single dataset.
    """
    saber = Saber(config=dummy_config_single_dataset)
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET)

    return saber

@pytest.fixture
def saber_single_dataset_embeddings(dummy_config_single_dataset):
    """Returns instance of `Saber` initialized with the dummy config file, a single dataset and
    embeddings.
    """
    saber = Saber(config=dummy_config_single_dataset)
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET)
    saber.load_embeddings(filepath=PATH_TO_DUMMY_EMBEDDINGS, binary=False)

    return saber

@pytest.fixture
def saber_single_dataset_model(dummy_config_single_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file, a single dataset
    a Keras model."""
    saber = Saber(config=dummy_config_single_dataset)
    saber.load_dataset(directory=PATH_TO_DUMMY_DATASET)
    saber.build()

    return saber

# COMPOUND DATASET

@pytest.fixture
def dummy_config_compound_dataset():
    """Returns an instance of a `Config` after parsing the dummy config file. Ensures that
    `replace_rare_tokens` argument is False. The compound dataset is just two copies of the dataset,
    this makes writing tests much simpler.
    """
    compound_dataset = [PATH_TO_DUMMY_DATASET, PATH_TO_DUMMY_DATASET]
    cli_arguments = {'replace_rare_tokens': False,
                     'dataset_folder': compound_dataset}
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def saber_compound_dataset(dummy_config_compound_dataset):
    """Returns an instance of `Saber` initialized with the dummy config file and a compound dataset.
    The compound dataset is just two copies of the dataset, this makes writing tests much
    simpler.
    """
    compound_dataset = [PATH_TO_DUMMY_DATASET, PATH_TO_DUMMY_DATASET]
    saber = Saber(config=dummy_config_compound_dataset)
    saber.load_dataset(directory=compound_dataset)

    return saber

############################################ UNIT TESTS ############################################

def test_attributes_after_initilization_of_model(saber_blank,
                                                 dummy_config_single_dataset):
    """Asserts instance attributes are initialized correctly when `Saber` object is created.
    """
    assert saber_blank.config is dummy_config_single_dataset

    assert saber_blank.preprocessor is None

    assert saber_blank.datasets is None
    assert saber_blank.embeddings is None
    assert saber_blank.model is None

    # test that we can pass arbitrary keyword arguments
    assert saber_blank.totally_arbitrary == 'arbitrary'

def test_load_dataset(saber_single_dataset):
    """Assert that the `datasets` attribute of a `Saber` instance was updated as expected after
    call to `Saber.load_dataset()`
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

def test_load_embeddings(saber_single_dataset_embeddings):
    """Assert that the `datasets` attribute of a `Saber` instance was updated as expected after
    call to `Saber.load_embeddings()`
    """
    assert isinstance(saber_single_dataset_embeddings.embeddings, Embeddings)

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

def test_build(saber_single_dataset_model):
    """Assert that the `model` attribute of a `Saber` instance was updated as expected after
    call to `Saber.build()`.
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

def test_annotate(saber_single_dataset_model):
    """Asserts that call to `Saber.annotate()` returns the expected results."""
    test = "This is a simple test. With multiple sentences"
    expected = {'text': test, 'ents': [], 'title': None}

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
