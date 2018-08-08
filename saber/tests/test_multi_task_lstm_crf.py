from os.path import abspath
import pytest

import numpy

from ..config import Config
from ..sequence_processor import SequenceProcessor

PATH_TO_DUMMY_CONFIG = abspath('saber/tests/resources/dummy_config.ini')
PATH_TO_DUMMY_DATASET = abspath('saber/tests/resources/dummy_dataset_1')
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = abspath('saber/tests/resources/dummy_word_embeddings/dummy_word_embeddings.txt')

# TODO (johngiorgi): fix some of the test_model_attributes_after_creation_of_model tests

@pytest.fixture
def dummy_config():
    """Returns an instance of a configparser object after parsing the dummy
    config file. """
    # create a dictionary to serve as cli arguments
    cli_arguments = {'dataset_folder': [PATH_TO_DUMMY_DATASET]}
    # create the config object, taking into account the CLI args
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def multi_task_lstm_crf_single_model(dummy_config):
    """Returns an instance of MultiTaskLSTMCRF initialized with the
    default configuration file and a single compiled model."""
    seq_processor_with_single_ds = SequenceProcessor(config=dummy_config)
    seq_processor_with_single_ds.load_dataset()
    seq_processor_with_single_ds.load_embeddings()
    seq_processor_with_single_ds.create_model()
    multi_task_lstm_crf_single_model = seq_processor_with_single_ds.model

    return multi_task_lstm_crf_single_model

def test_model_attributes_after_creation_of_model(multi_task_lstm_crf_single_model, dummy_config):
    """Asserts instance attributes are initialized correctly after the loading of a dataset
    and the creation of a model."""
    # MultiTaskLSTMCRF object attributes
    assert multi_task_lstm_crf_single_model.config is dummy_config
    assert isinstance(multi_task_lstm_crf_single_model.token_embedding_matrix, numpy.ndarray)

    # Attributes of Config object tied to MultiTaskLSTMCRF instance
    assert multi_task_lstm_crf_single_model.config.activation == 'relu'
    assert multi_task_lstm_crf_single_model.config.batch_size == 32
    assert multi_task_lstm_crf_single_model.config.char_embed_dim == 30
    assert multi_task_lstm_crf_single_model.config.criteria == 'exact'
    assert multi_task_lstm_crf_single_model.config.dataset_folder == [PATH_TO_DUMMY_DATASET]
    assert not multi_task_lstm_crf_single_model.config.debug
    assert multi_task_lstm_crf_single_model.config.decay == 0.0
    assert multi_task_lstm_crf_single_model.config.dropout_rate == {'input': 0.3, 'output':0.3, 'recurrent': 0.1}
    assert not multi_task_lstm_crf_single_model.config.fine_tune_word_embeddings
    assert multi_task_lstm_crf_single_model.config.grad_norm == 1.0
    assert multi_task_lstm_crf_single_model.config.k_folds == 2
    assert multi_task_lstm_crf_single_model.config.learning_rate == 0.0
    assert multi_task_lstm_crf_single_model.config.epochs == 50
    assert multi_task_lstm_crf_single_model.config.model_name == 'mt-lstm-crf'
    assert multi_task_lstm_crf_single_model.config.optimizer == 'nadam'
    assert multi_task_lstm_crf_single_model.config.output_folder == abspath('../output')
    assert multi_task_lstm_crf_single_model.config.pretrained_model_weights == ''
    assert not multi_task_lstm_crf_single_model.config.replace_rare_tokens
    assert multi_task_lstm_crf_single_model.config.word_embed_dim == 200
    assert multi_task_lstm_crf_single_model.config.pretrained_embeddings == PATH_TO_DUMMY_TOKEN_EMBEDDINGS
    assert multi_task_lstm_crf_single_model.config.train_model
    assert not multi_task_lstm_crf_single_model.config.verbose
