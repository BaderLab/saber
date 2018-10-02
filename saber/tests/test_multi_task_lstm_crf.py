import os

import numpy

import pytest

from ..config import Config
from ..constants import (PATH_TO_DUMMY_CONFIG, PATH_TO_DUMMY_DATASET,
                         PATH_TO_DUMMY_EMBEDDINGS)
from ..sequence_processor import SequenceProcessor

# TODO (johngiorgi): fix some of the test_model_attributes_after_creation_of_model tests

@pytest.fixture
def dummy_config():
    """Returns an instance of a configparser object after parsing the dummy config file."""
    # the dataset and embeddings are used for test purposes so they must point to the
    # correct resources, this can be ensured by passing their paths here
    cli_arguments = {'dataset_folder': [PATH_TO_DUMMY_DATASET],
                     'pretrained_embeddings': PATH_TO_DUMMY_EMBEDDINGS}
    # create the config object, taking into account the CLI args
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    dummy_config._process_args(cli_arguments)

    return dummy_config

@pytest.fixture
def multi_task_lstm_crf_single_model(dummy_config):
    """Returns an instance of MultiTaskLSTMCRF initialized with the default configuration file and
    a single compiled model."""
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
    # TEMP: need a better solution than this
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
    assert multi_task_lstm_crf_single_model.config.output_folder == os.path.abspath('../output')
    assert multi_task_lstm_crf_single_model.config.pretrained_model_weights == ''
    assert not multi_task_lstm_crf_single_model.config.replace_rare_tokens
    assert multi_task_lstm_crf_single_model.config.word_embed_dim == 200
    # TEMP: need a better solution than this
    assert multi_task_lstm_crf_single_model.config.pretrained_embeddings == PATH_TO_DUMMY_EMBEDDINGS
    assert multi_task_lstm_crf_single_model.config.train_model
    assert not multi_task_lstm_crf_single_model.config.verbose
