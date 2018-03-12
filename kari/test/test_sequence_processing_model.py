import numpy
import pytest

from utils_parameter_parsing import *
from sequence_processor import SequenceProcessor

# constants for dummy dataset/config/word embeddings to perform testing on
PATH_TO_DUMMY_CONFIG = 'kari/test/resources/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'kari/test/resources/single_dummy_dataset'
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = 'kari/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt'
DUMMY_TRAIN_SENT_NUM = 2
DUMMY_TEST_SENT_NUM = 1
# embedding matrix shape is num word types x dimension of embeddings
DUMMY_EMBEDDINGS_MATRIX_SHAPE = (25, 2)

@pytest.fixture
def dummy_config():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # parse the dummy config
    dummy_config = config_parser(PATH_TO_DUMMY_CONFIG)

    return dummy_config

@pytest.fixture
def model_without_dataset(dummy_config):
    """Returns an instance of SequenceProcessingModel initialized with the
    default configuration file and no loaded dataset. """
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config)

    model_without_dataset = SequenceProcessor(config=parameters)

    return model_without_dataset

@pytest.fixture
def model_with_single_dataset(dummy_config):
    """Returns an instance of SequenceProcessingModel initialized with the
    default configuration file and a single loaded dataset. """
    # create a dictionary to serve as cli arguments
    cli_arguments = {'dataset_folder': PATH_TO_DUMMY_DATASET}
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config, cli_arguments)

    model_with_single_dataset = SequenceProcessor(config=parameters)
    model_with_single_dataset.load_dataset()

    return model_with_single_dataset

@pytest.fixture
def model_with_compound_dataset(dummy_config):
    """Returns an instance of SequenceProcessingModel initialized with the
    default configuration file and a compound loaded dataset. The compound
    dataset is just two copies of the dataset, this makes writing tests
    much simpler. """
    # create a dictionary to serve as cli arguments
    compound_dataset = PATH_TO_DUMMY_DATASET + ',' + PATH_TO_DUMMY_DATASET
    cli_arguments = {'dataset_folder': compound_dataset}
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config, cli_arguments)

    model_with_compound_dataset = SequenceProcessor(config=parameters)
    model_with_compound_dataset.load_dataset()

    return model_with_compound_dataset

def test_attributes_after_initilization_of_model(model_without_dataset):
    """Asserts instance attributes are initialized correctly when sequence
    model is initialized (and before dataset is loaded)."""
    # check value/type
    assert model_without_dataset.config['activation_function'] == 'relu'
    assert model_without_dataset.config['batch_size'] == 1
    assert model_without_dataset.config['character_embedding_dimension'] == 30
    assert model_without_dataset.config['dataset_folder'][0] == PATH_TO_DUMMY_DATASET
    assert model_without_dataset.config['debug'] == False
    assert model_without_dataset.config['dropout_rate'] == 0.3
    assert model_without_dataset.config['freeze_token_embeddings'] == True
    assert model_without_dataset.config['gradient_clipping_value'] == 0.0
    assert model_without_dataset.config['k_folds'] == 5
    assert model_without_dataset.config['learning_rate'] == 0.01
    assert model_without_dataset.config['load_pretrained_model'] == False
    assert model_without_dataset.config['maximum_number_of_epochs'] == 10
    assert model_without_dataset.config['model_name'] == 'MT-LSTM-CRF'
    assert model_without_dataset.config['optimizer'] == 'sgd'
    assert model_without_dataset.config['output_folder'] == '../output'
    assert model_without_dataset.config['pretrained_model_weights'] == ''
    assert model_without_dataset.config['token_embedding_dimension'] == 200
    assert model_without_dataset.config['token_pretrained_embedding_filepath'] == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert model_without_dataset.config['train_model'] == True
    assert model_without_dataset.config['max_seq_len'] == 50

    assert model_without_dataset.ds == []
    assert model_without_dataset.token_embedding_matrix == None
    assert model_without_dataset.model == []

def test_X_input_sequences_after_loading_single_dataset(model_with_single_dataset):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = model_with_single_dataset.ds[0]
    model = model_with_single_dataset
    # check type
    assert type(ds.train_word_idx_sequence) == numpy.ndarray
    # check shape
    assert ds.train_word_idx_sequence.shape == (DUMMY_TRAIN_SENT_NUM, model.config['max_seq_len'])

def test_y_output_sequences_after_loading_single_dataset(model_with_single_dataset):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = model_with_single_dataset.ds[0]
    model = model_with_single_dataset
    # check type
    assert type(ds.train_tag_idx_sequence) == numpy.ndarray
    # check value
    assert ds.train_tag_idx_sequence.shape == (DUMMY_TRAIN_SENT_NUM,
        ds.max_seq_len, ds.tag_type_count)

def test_word_embeddings_after_loading_single_dataset(model_with_single_dataset):
    """Asserts that pretained token embeddings are loaded correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    model = model_with_single_dataset
    # check type
    assert type(model.token_embedding_matrix) == numpy.ndarray
    # check value
    assert model.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE

def test_agreement_between_model_and_single_dataset(model_with_single_dataset):
    """Asserts that the attributes common to SequenceProcessingModel and
    Dataset are the same for single datasets."""
    # shortens assert statments
    ds = model_with_single_dataset.ds[0]
    model = model_with_single_dataset

    assert model.config['dataset_folder'][0] == ds.dataset_folder
    assert model.config['max_seq_len'] == ds.max_seq_len

def test_X_input_sequences_after_loading_compound_dataset(model_with_compound_dataset):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in model_with_compound_dataset.ds:
        # check type
        assert type(ds.train_word_idx_sequence) == numpy.ndarray
        # check shape
        assert ds.train_word_idx_sequence.shape == (DUMMY_TRAIN_SENT_NUM,
                                                    ds.max_seq_len)

def test_y_output_sequences_after_loading_compound_dataset(model_with_compound_dataset):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in model_with_compound_dataset.ds:
        assert type(ds.train_tag_idx_sequence) == numpy.ndarray
        # check value
        assert ds.train_tag_idx_sequence.shape == (DUMMY_TRAIN_SENT_NUM,
                                                   ds.max_seq_len,
                                                   ds.tag_type_count)

def test_word_embeddings_after_loading_compound_dataset(model_with_compound_dataset):
    """ Asserts that pretained token embeddings are loaded correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets """
    # shortens assert statments
    model = model_with_compound_dataset
    # check type
    assert type(model.token_embedding_matrix) == numpy.ndarray
    # check value
    assert model.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE

def test_agreement_between_model_and_compound_dataset(model_with_compound_dataset):
    """ Asserts that the attributes common to SequenceProcessingModel and
    Dataset are the same for compound datasets.
    """
    # shortens assert statments
    model = model_with_compound_dataset
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for i, ds in enumerate(model.ds):
        assert model.config['dataset_folder'][i] == ds.dataset_folder
        assert model.config['max_seq_len'] == ds.max_seq_len
