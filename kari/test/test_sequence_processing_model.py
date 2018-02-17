import pytest

from utils_parameter_parsing import *
from sequence_processing_model import SequenceProcessingModel

PATH_TO_DUMMY_CONFIG = 'kari/test/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'kari/test/dummy_dataset'

@pytest.fixture
def default_model():
    """ Returns an instance of SequenceProcessingModel initialized with the
    default configuration file. """

    cli_arguments = parse_arguments() # parse CL args
    config = config_parser(PATH_TO_DUMMY_CONFIG) # parse config.ini
    # resolve parameters, cast to correct types
    parameters = process_parameters(config, cli_arguments)

    default_model = SequenceProcessingModel(**parameters)

    return default_model

def test_attributes_after_initilization_of_model(default_model):
    """ Asserts instance attributes are initialized correctly when sequence
    model is initialized. """

    # check type
    assert type(default_model.config_filepath) == str
    assert type(default_model.dataset_text_folder) == str
    assert type(default_model.debug) == bool
    assert type(default_model.dropout_rate) == float
    assert type(default_model.gradient_clipping_value) == float
    assert type(default_model.k_folds) == int
    assert type(default_model.learning_rate) == float
    assert type(default_model.maximum_number_of_epochs) == int
    assert type(default_model.optimizer) == str
    assert type(default_model.output_folder) == str
    assert type(default_model.train_model) == bool
    assert type(default_model.max_seq_len) == int
    # check value
    assert default_model.config_filepath == './config.ini'
    assert default_model.dataset_text_folder == PATH_TO_DUMMY_DATASET
    assert default_model.debug == False
    assert default_model.dropout_rate == 0.5
    assert default_model.gradient_clipping_value == 5.0
    assert default_model.k_folds == 5
    assert default_model.learning_rate == 0.005
    assert default_model.maximum_number_of_epochs == 10
    assert default_model.optimizer == 'sgd'
    assert default_model.output_folder == '../output'
    assert default_model.train_model == True
    assert default_model.max_seq_len == 100
