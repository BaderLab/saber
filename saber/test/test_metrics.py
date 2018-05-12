import pytest

from utils_models import *
from utils_parameter_parsing import *
from metrics import Metrics
from sequence_processor import SequenceProcessor

PATH_TO_DUMMY_CONFIG = 'saber/test/resources/dummy_config.ini'
PATH_TO_DUMMY_DATASET = 'saber/test/resources/dummy_dataset'
PATH_TO_METRICS_OUTPUT = 'totally/arbitrary'

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

@pytest.fixture
def train_valid_indices_single_model(multi_task_lstm_crf_single_model):
    """Returns an train/valid indices from call to get_train_valid_indices()
    of a MultiTaskLSTMCRF object."""
    ds_ = multi_task_lstm_crf_single_model.ds
    k_folds = multi_task_lstm_crf_single_model.config['k_folds']

    train_valid_indices_single_model = get_train_valid_indices(ds_, k_folds)
    return train_valid_indices_single_model

@pytest.fixture
def data_partitions_single_model(multi_task_lstm_crf_single_model, train_valid_indices_single_model):
    """Returns a data partitions from call to _get_data_partitions()
    of a MultiTaskLSTMCRF object for fold=0."""
    ds_ = multi_task_lstm_crf_single_model.ds

    data_partitions_single_model = get_data_partitions(ds_, train_valid_indices_single_model, fold=0)
    return data_partitions_single_model

@pytest.fixture
def metrics_single_model(multi_task_lstm_crf_single_model, data_partitions_single_model):
    """Returns list of Metrics objects returned from call to get_metrics()"""
    model_ = multi_task_lstm_crf_single_model
    ds_ = model_.ds

    return get_metrics(ds_, data_partitions_single_model, [PATH_TO_METRICS_OUTPUT])

def test_attributes_after_initilization_of_metrics(multi_task_lstm_crf_single_model,
                                                   data_partitions_single_model,
                                                   metrics_single_model):
    """
    """
    # single dataset, so the ith index of data_partitions_single_model and
    # metrics_single_model is 0
    idx = 0
    model_ = multi_task_lstm_crf_single_model
    ds_ = model_.ds[idx]
    metrics_ = metrics_single_model[idx]

    assert metrics_.X_train[0].shape == data_partitions_single_model[idx][0].shape
    assert metrics_.X_train[-1].shape == data_partitions_single_model[idx][2].shape
    assert metrics_.X_valid[0].shape == data_partitions_single_model[idx][1].shape
    assert metrics_.X_valid[-1].shape == data_partitions_single_model[idx][3].shape
    assert metrics_.y_train.shape == data_partitions_single_model[idx][4].shape
    assert metrics_.y_valid.shape == data_partitions_single_model[idx][5].shape

    assert metrics_.tag_type_to_idx == ds_.tag_type_to_idx
    assert metrics_.output_dir == PATH_TO_METRICS_OUTPUT
    assert metrics_.current_epoch == 0

    assert metrics_.train_performance_metrics_per_epoch == []
    assert metrics_.valid_performance_metrics_per_epoch == []
