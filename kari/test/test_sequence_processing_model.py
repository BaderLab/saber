import numpy
import pytest

from utils_parameter_parsing import *
from sequence_processing_model import SequenceProcessingModel

PATH_TO_DUMMY_CONFIG = 'kari/test/resources/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'kari/test/resources/dummy_dataset'
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = 'kari/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt'
DUMMY_TRAIN_SENT_NUM = 2
DUMMY_TEST_SENT_NUM = 1
# embedding matrix shape is num word types x dimension of embeddings
DUMMY_EMBEDDINGS_MATRIX_SHAPE = (28, 2)

@pytest.fixture
def model_without_dataset():
    """ Returns an instance of SequenceProcessingModel initialized with the
    default configuration file and no loaded dataset"""
    config = config_parser(PATH_TO_DUMMY_CONFIG) # parse config.ini
    # create a dictionary to serve as cli arguments
    cli_arguments = {}
    # resolve parameters, cast to correct types
    parameters = process_parameters(config, cli_arguments)

    model_without_dataset = SequenceProcessingModel(**parameters)

    return model_without_dataset

@pytest.fixture
def model_with_dataset():
    """ Returns an instance of SequenceProcessingModel initialized with the
    default configuration file and a loaded dataset """
    config = config_parser(PATH_TO_DUMMY_CONFIG) # parse config.ini
    # create a dictionary to serve as cli arguments
    cli_arguments = {'dataset_text_folder': PATH_TO_DUMMY_DATASET}
    # resolve parameters, cast to correct types
    parameters = process_parameters(config, cli_arguments)

    model_with_dataset = SequenceProcessingModel(**parameters)
    model_with_dataset.load_dataset()

    return model_with_dataset

def test_attributes_after_initilization_of_model(model_without_dataset):
    """ Asserts instance attributes are initialized correctly when sequence
    model is initialized (and before dataset is loaded)."""
    # check type
    assert type(model_without_dataset.activation_function) == str
    assert type(model_without_dataset.batch_size) == int
    assert type(model_without_dataset.dataset_text_folder) == str
    assert type(model_without_dataset.debug) == bool
    assert type(model_without_dataset.dropout_rate) == float
    assert type(model_without_dataset.freeze_token_embeddings) == bool
    assert type(model_without_dataset.gradient_clipping_value) == float
    assert type(model_without_dataset.k_folds) == int
    assert type(model_without_dataset.learning_rate) == float
    assert type(model_without_dataset.load_pretrained_model) == bool
    assert type(model_without_dataset.maximum_number_of_epochs) == int
    assert type(model_without_dataset.model_name) == str
    assert type(model_without_dataset.optimizer) == str
    assert type(model_without_dataset.output_folder) == str
    assert type(model_without_dataset.pretrained_model_weights) == str
    assert type(model_without_dataset.token_pretrained_embedding_filepath) == str
    assert type(model_without_dataset.train_model) == bool
    assert type(model_without_dataset.max_seq_len) == int
    # check value
    assert model_without_dataset.activation_function == 'relu'
    assert model_without_dataset.batch_size == 1
    assert model_without_dataset.dataset_text_folder == PATH_TO_DUMMY_DATASET
    assert model_without_dataset.debug == False
    assert model_without_dataset.dropout_rate == 0.1
    assert model_without_dataset.freeze_token_embeddings == True
    assert model_without_dataset.gradient_clipping_value == 0.0
    assert model_without_dataset.k_folds == 5
    assert model_without_dataset.learning_rate == 0.01
    assert model_without_dataset.load_pretrained_model == False
    assert model_without_dataset.maximum_number_of_epochs == 10
    assert model_without_dataset.model_name == 'LSTM-CRF-NER'
    assert model_without_dataset.optimizer == 'sgd'
    assert model_without_dataset.output_folder == '../output'
    assert model_without_dataset.pretrained_model_weights == ''
    assert model_without_dataset.token_pretrained_embedding_filepath == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert model_without_dataset.train_model == True
    assert model_without_dataset.max_seq_len == 50

def test_X_input_sequences_after_loading_dataset(model_with_dataset):
    """ Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded). """
    assert type(model_with_dataset.X_train) == numpy.ndarray
    assert type(model_with_dataset.X_test) == numpy.ndarray

    assert model_with_dataset.X_train.shape == (DUMMY_TRAIN_SENT_NUM, model_with_dataset.max_seq_len)
    assert model_with_dataset.X_test.shape == (DUMMY_TEST_SENT_NUM, model_with_dataset.max_seq_len)

def test_y_output_sequences_after_loading_dataset(model_with_dataset):
    """ Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded). """
    assert type(model_with_dataset.y_train) == numpy.ndarray
    assert type(model_with_dataset.y_test) == numpy.ndarray
    # check value
    assert model_with_dataset.y_train.shape == (DUMMY_TRAIN_SENT_NUM,
        model_with_dataset.max_seq_len, model_with_dataset.ds.tag_type_count)
    assert model_with_dataset.y_test.shape == (DUMMY_TEST_SENT_NUM,
        model_with_dataset.max_seq_len, model_with_dataset.ds.tag_type_count)

def test_word_embeddings_after_loading_dataset(model_with_dataset):
    """ Asserts that pretained token embeddings are loaded correctly when
    sequence model is initialized (and after dataset is loaded). """
    # check type
    assert type(model_with_dataset.token_embedding_matrix) == numpy.ndarray
    # check value
    assert model_with_dataset.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE
