import pytest
import numpy

from utils_parameter_parsing import *
from dataset import Dataset
from sequence_processor import SequenceProcessor

PATH_TO_DUMMY_CONFIG = 'saber/test/resources/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'saber/test/resources/dummy_dataset'
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = 'saber/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt'

# TODO (johngiorgi): fix some of the test_model_attributes_after_creation_of_model tests

@pytest.fixture
def dummy_config():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # parse the dummy config
    dummy_config = config_parser(PATH_TO_DUMMY_CONFIG)

    return dummy_config

@pytest.fixture
def multi_task_lstm_crf_single_model(dummy_config):
    """Returns an instance of MultiTaskLSTMCRF initialized with the
    default configuration file and a single compiled model."""
    # create a dictionary to serve as cli arguments
    cli_arguments = {'dataset_folder': [PATH_TO_DUMMY_DATASET]}
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config, cli_arguments)

    seq_processor_with_single_ds = SequenceProcessor(config=parameters)
    seq_processor_with_single_ds.load_dataset()
    seq_processor_with_single_ds.load_embeddings()
    seq_processor_with_single_ds.create_model()
    multi_task_lstm_crf_single_model = seq_processor_with_single_ds.model

    return multi_task_lstm_crf_single_model

def test_model_attributes_after_creation_of_model(multi_task_lstm_crf_single_model):
    """Asserts instance attributes are initialized correctly when sequence
    model is initialized (and before dataset is loaded)."""
    # check value/type
    assert multi_task_lstm_crf_single_model.config['activation_function'] == 'relu'
    assert multi_task_lstm_crf_single_model.config['batch_size'] == 1
    assert multi_task_lstm_crf_single_model.config['character_embedding_dimension'] == 30
    assert multi_task_lstm_crf_single_model.config['dataset_folder'] == [PATH_TO_DUMMY_DATASET]
    assert multi_task_lstm_crf_single_model.config['debug'] == False
    assert multi_task_lstm_crf_single_model.config['dropout_rate'] == 0.3
    assert multi_task_lstm_crf_single_model.config['freeze_token_embeddings'] == True
    assert multi_task_lstm_crf_single_model.config['gradient_normalization'] == None
    assert multi_task_lstm_crf_single_model.config['k_folds'] == 2
    assert multi_task_lstm_crf_single_model.config['learning_rate'] == 0.01
    assert multi_task_lstm_crf_single_model.config['decay'] == 0.05
    assert multi_task_lstm_crf_single_model.config['maximum_number_of_epochs'] == 10
    assert multi_task_lstm_crf_single_model.config['optimizer'] == 'sgd'
    assert multi_task_lstm_crf_single_model.config['output_folder'] == '../output'
    assert multi_task_lstm_crf_single_model.config['pretrained_model_weights'] == ''
    assert multi_task_lstm_crf_single_model.config['token_embedding_dimension'] == 200
    assert multi_task_lstm_crf_single_model.config['token_pretrained_embedding_filepath'] == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert multi_task_lstm_crf_single_model.config['verbose'] == False

    # assert type(multi_task_lstm_crf_single_model.ds) == []
    assert type(multi_task_lstm_crf_single_model.token_embedding_matrix) == numpy.ndarray
    # assert type(multi_task_lstm_crf_single_model.model) == []
    # assert len(multi_task_lstm_crf_single_model.model) == 1
