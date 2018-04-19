import numpy
import pytest

from utils_parameter_parsing import *
from sequence_processor import SequenceProcessor

# constants for dummy dataset/config/word embeddings to perform testing on
PATH_TO_DUMMY_CONFIG = 'saber/test/resources/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'saber/test/resources/single_dummy_dataset'
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = 'saber/test/resources/dummy_word_embeddings/dummy_word_embeddings.txt'
DUMMY_TRAIN_SENT_NUM = 2
DUMMY_TEST_SENT_NUM = 1
DUMMY_TAG_TYPE_COUNT = 5
# embedding matrix shape is num word types x dimension of embeddings
DUMMY_EMBEDDINGS_MATRIX_SHAPE = (25, 2)

# TODO (johngiorgi): add some kind of test that accounts for the error thrown
# when we try to load token embedding before loading a dataset.

@pytest.fixture
def dummy_config():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # parse the dummy config
    dummy_config = config_parser(PATH_TO_DUMMY_CONFIG)

    return dummy_config

@pytest.fixture
def model_no_ds_no_embed(dummy_config):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file and no loaded dataset. """
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config)

    model_no_ds_no_embed = SequenceProcessor(config=parameters)

    return model_no_ds_no_embed

@pytest.fixture
def model_sing_ds_no_embed(dummy_config):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file and a single loaded dataset. """
    # create a dictionary to serve as cli arguments
    cli_arguments = {'dataset_folder': [PATH_TO_DUMMY_DATASET]}
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config, cli_arguments)

    model_sing_ds_no_embed = SequenceProcessor(config=parameters)
    model_sing_ds_no_embed.load_dataset()

    return model_sing_ds_no_embed

@pytest.fixture
def model_compound_ds_no_embed(dummy_config):
    """Returns an instance of SequenceProcessor initialized with the
    default configuration file and a compound loaded dataset. The compound
    dataset is just two copies of the dataset, this makes writing tests
    much simpler. """
    # create a dictionary to serve as cli arguments
    compound_dataset = [PATH_TO_DUMMY_DATASET, PATH_TO_DUMMY_DATASET]
    cli_arguments = {'dataset_folder': compound_dataset}
    # resolve parameters, cast to correct types
    parameters = process_parameters(dummy_config, cli_arguments)

    model_compound_ds_no_embed = SequenceProcessor(config=parameters)
    model_compound_ds_no_embed.load_dataset()

    return model_compound_ds_no_embed

def test_attributes_after_initilization_of_model(model_no_ds_no_embed):
    """Asserts instance attributes are initialized correctly when sequence
    model is initialized (and before dataset is loaded)."""
    # check value/type
    assert model_no_ds_no_embed.config['activation_function'] == 'relu'
    assert model_no_ds_no_embed.config['batch_size'] == 1
    assert model_no_ds_no_embed.config['character_embedding_dimension'] == 30
    assert model_no_ds_no_embed.config['dataset_folder'] == [PATH_TO_DUMMY_DATASET]
    assert model_no_ds_no_embed.config['debug'] == False
    assert model_no_ds_no_embed.config['dropout_rate'] == 0.3
    assert model_no_ds_no_embed.config['freeze_token_embeddings'] == True
    assert model_no_ds_no_embed.config['gradient_normalization'] == None
    assert model_no_ds_no_embed.config['k_folds'] == 2
    assert model_no_ds_no_embed.config['learning_rate'] == 0.01
    assert model_no_ds_no_embed.config['decay'] == 0.05
    assert model_no_ds_no_embed.config['load_pretrained_model'] == False
    assert model_no_ds_no_embed.config['maximum_number_of_epochs'] == 10
    assert model_no_ds_no_embed.config['model_name'] == 'MT-LSTM-CRF'
    assert model_no_ds_no_embed.config['optimizer'] == 'sgd'
    assert model_no_ds_no_embed.config['output_folder'] == '../output'
    assert model_no_ds_no_embed.config['pretrained_model_weights'] == ''
    assert model_no_ds_no_embed.config['token_embedding_dimension'] == 200
    assert model_no_ds_no_embed.config['token_pretrained_embedding_filepath'] == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert model_no_ds_no_embed.config['train_model'] == True
    assert model_no_ds_no_embed.config['verbose'] == False

    assert model_no_ds_no_embed.ds == []
    assert model_no_ds_no_embed.token_embedding_matrix == None
    assert model_no_ds_no_embed.model == None

def test_token_embeddings_load(model_sing_ds_no_embed,
                               model_compound_ds_no_embed):
    """Asserts that pre-trained token embeddings are loaded correctly when
    SequenceProcessor.load_embeddings() is called"""
    # load embeddings for each model
    model_sing_ds_no_embed.load_embeddings()
    model_compound_ds_no_embed.load_embeddings()

    # check type
    assert type(model_sing_ds_no_embed.token_embedding_matrix) == numpy.ndarray
    assert type(model_compound_ds_no_embed.token_embedding_matrix) == numpy.ndarray
    # check value
    assert model_sing_ds_no_embed.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE
    assert model_compound_ds_no_embed.token_embedding_matrix.shape == DUMMY_EMBEDDINGS_MATRIX_SHAPE

def test_X_input_sequences_after_loading_single_dataset(model_sing_ds_no_embed):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = model_sing_ds_no_embed.ds[0]
    model = model_sing_ds_no_embed
    # check type
    assert type(ds.train_word_idx_seq) == numpy.ndarray
    # check shape
    assert ds.train_word_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM

def test_y_output_sequences_after_loading_single_dataset(model_sing_ds_no_embed):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for single
    datasets."""
    # shortens assert statments
    ds = model_sing_ds_no_embed.ds[0]
    model = model_sing_ds_no_embed
    # check type
    assert type(ds.train_tag_idx_seq) == numpy.ndarray
    # check value
    assert ds.train_tag_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM
    assert ds.train_tag_idx_seq.shape[-1] == DUMMY_TAG_TYPE_COUNT

def test_agreement_between_model_and_single_dataset(model_sing_ds_no_embed):
    """Asserts that the attributes common to SequenceProcessor and
    Dataset are the same for single datasets."""
    # shortens assert statments
    ds = model_sing_ds_no_embed.ds[0]
    model = model_sing_ds_no_embed

    assert model.config['dataset_folder'][0] == ds.filepath

def test_X_input_sequences_after_loading_compound_dataset(model_compound_ds_no_embed):
    """Asserts X (input) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in model_compound_ds_no_embed.ds:
        # check type
        assert type(ds.train_word_idx_seq) == numpy.ndarray
        # check shape
        assert ds.train_word_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM

def test_y_output_sequences_after_loading_compound_dataset(model_compound_ds_no_embed):
    """Asserts y (labels) data partition attribute is initialized correctly when
    sequence model is initialized (and after dataset is loaded) for compound
    datasets."""
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for ds in model_compound_ds_no_embed.ds:
        assert type(ds.train_tag_idx_seq) == numpy.ndarray
        # check value
        assert ds.train_tag_idx_seq.shape[0] == DUMMY_TRAIN_SENT_NUM
        assert ds.train_tag_idx_seq.shape[-1] == DUMMY_TAG_TYPE_COUNT

def test_agreement_between_model_and_compound_dataset(model_compound_ds_no_embed):
    """Asserts that the attributes common to SequenceProcessor and
    Dataset are the same for compound datasets.
    """
    # shortens assert statments
    model = model_compound_ds_no_embed
    # for testing purposes, the datasets are identical so we can simply peform
    # the same checks for each in a loop
    for i, ds in enumerate(model.ds):
        assert model.config['dataset_folder'][i] == ds.filepath
