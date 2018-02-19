import numpy
import pytest

from utils_parameter_parsing import *
from sequence_processing_model import SequenceProcessingModel

PATH_TO_DUMMY_CONFIG = 'kari/test/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'kari/test/dummy_dataset'
DUMMY_TRAIN_SENT_NUM = 2
DUMMY_TEST_SENT_NUM = 1

@pytest.fixture
def default_model():
    """ Returns an instance of SequenceProcessingModel initialized with the
    default configuration file. """

    config = config_parser(PATH_TO_DUMMY_CONFIG) # parse config.ini
    # create a dictionary to serve as cli arguments
    cli_arguments = {'dataset_text_folder': PATH_TO_DUMMY_DATASET}
    # resolve parameters, cast to correct types
    parameters = process_parameters(config, cli_arguments)

    default_model = SequenceProcessingModel(**parameters)

    return default_model

def test_attributes_after_initilization_of_model(default_model):
    """ Asserts instance attributes are initialized correctly when sequence
    model is initialized. """
    # check type
    assert type(default_model.activation_function) == str
    assert type(default_model.batch_size) == int
    assert type(default_model.dataset_text_folder) == str
    assert type(default_model.debug) == bool
    assert type(default_model.dropout_rate) == float
    assert type(default_model.freeze_token_embeddings) == bool
    assert type(default_model.gradient_clipping_value) == float
    assert type(default_model.k_folds) == int
    assert type(default_model.learning_rate) == float
    assert type(default_model.maximum_number_of_epochs) == int
    assert type(default_model.model_name) == str
    assert type(default_model.optimizer) == str
    assert type(default_model.output_folder) == str
    assert type(default_model.token_pretrained_embedding_filepath) == str
    assert type(default_model.train_model) == bool
    assert type(default_model.max_seq_len) == int
    # check value
    assert default_model.activation_function == 'relu'
    assert default_model.batch_size == 1
    assert default_model.dataset_text_folder == PATH_TO_DUMMY_DATASET
    assert default_model.debug == False
    assert default_model.dropout_rate == 0.1
    assert default_model.freeze_token_embeddings == True
    assert default_model.gradient_clipping_value == 0.0
    assert default_model.k_folds == 5
    assert default_model.learning_rate == 0.01
    assert default_model.maximum_number_of_epochs == 10
    assert default_model.model_name == 'LSTM-CRF-NER'
    assert default_model.optimizer == 'sgd'
    assert default_model.output_folder == '../output'
    assert default_model.token_pretrained_embedding_filepath == ''
    assert default_model.train_model == True
    assert default_model.max_seq_len == 50

def test_X_input_sequences_after_initilization_of_model(default_model):
    """ Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized. """
    assert type(default_model.X_train) == numpy.ndarray
    assert type(default_model.X_test) == numpy.ndarray

    assert default_model.X_train.shape == (DUMMY_TRAIN_SENT_NUM, default_model.max_seq_len)
    assert default_model.X_test.shape == (DUMMY_TEST_SENT_NUM, default_model.max_seq_len)

def test_y_output_sequences_after_initilization_of_model(default_model):
    """ Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized. """
    assert type(default_model.y_train) == numpy.ndarray
    assert type(default_model.y_test) == numpy.ndarray

    assert default_model.y_train.shape == (DUMMY_TRAIN_SENT_NUM,
        default_model.max_seq_len, default_model.ds.tag_type_count)
    assert default_model.y_test.shape == (DUMMY_TEST_SENT_NUM,
        default_model.max_seq_len, default_model.ds.tag_type_count)
