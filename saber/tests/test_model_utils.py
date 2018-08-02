"""Any and all unit tests for the model_utils (saber/utils/model_utils.py).
"""
import os
import pytest

from ..utils import model_utils
from ..metrics import Metrics
from ..config import Config
from ..sequence_processor import SequenceProcessor

PATH_TO_DUMMY_CONFIG = os.path.abspath('saber/tests/resources/dummy_config.ini')
PATH_TO_DUMMY_DATASET = os.path.abspath('saber/tests/resources/dummy_dataset_1')
PATH_TO_DUMMY_TOKEN_EMBEDDINGS = os.path.abspath(('saber/tests/resources/dummy_word_embeddings/'
                                                  'dummy_word_embeddings.txt'))

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

    return seq_processor_with_single_ds.model

@pytest.fixture
def training_data_single_model(multi_task_lstm_crf_single_model):
    """Returns a list of dictionaries, one for each dataset, containing the training dataself.
    Dictionary contains keys 'X_train', 'X_valid', 'X_test', 'y_train', 'y_valid', 'y_test'"""
    datasets = multi_task_lstm_crf_single_model.ds

    training_data = {}
    for i, ds in enumerate(datasets):
        training_data[i] = {'X_train': [ds.idx_seq['train']['word'], ds.idx_seq['train']['char']],
                            'X_valid': None,
                            'X_test': None,
                            'y_train': ds.idx_seq['train']['tag'],
                            'y_valid': None,
                            'y_test': None}
    return training_data

@pytest.fixture
def train_valid_indices_single_model(multi_task_lstm_crf_single_model, training_data_single_model):
    """Returns an train/valid indices from call to get_train_valid_indices() of a MultiTaskLSTMCRF
    object."""
    datasets = multi_task_lstm_crf_single_model.ds
    k_folds = multi_task_lstm_crf_single_model.config.k_folds

    return model_utils.get_train_valid_indices(training_data_single_model, k_folds)

@pytest.fixture
def data_partitions_single_model(multi_task_lstm_crf_single_model, train_valid_indices_single_model,
                                 training_data_single_model):
    """Returns a data partitions from call to _get_data_partitions()
    of a MultiTaskLSTMCRF object for fold=0."""
    datasets = multi_task_lstm_crf_single_model.ds

    data_partitions = model_utils.get_data_partitions(training_data_single_model,
                                                      train_valid_indices_single_model,
                                                      fold=0)
    return data_partitions

@pytest.fixture
def metrics_single_model(multi_task_lstm_crf_single_model, training_data_single_model):
    """Returns list of Metrics objects returned from call to get_metrics()"""
    model_ = multi_task_lstm_crf_single_model
    datasets = model_.ds

    # create the output folder for metrics object
    train_session_dir = model_utils.prepare_output_directory(model_.config.dataset_folder,
                                                             model_.config.output_folder)

    return model_utils.get_metrics(datasets=datasets,
                                   training_data=training_data_single_model,
                                   output_dir=train_session_dir)

def test_precision_recall_f1_support():
    """Asserts that model_utils.precision_recall_f1_support returns the expected values."""
    TP_dummy = 100
    FP_dummy = 10
    FN_dummy = 20

    prec_dummy = TP_dummy / (TP_dummy + FP_dummy)
    rec_dummy = TP_dummy / (TP_dummy + FN_dummy)
    f1_dummy = 2 * prec_dummy * rec_dummy / (prec_dummy + rec_dummy)
    support_dummy = TP_dummy + FN_dummy

    test_scores_no_null = model_utils.precision_recall_f1_support(TP_dummy, FP_dummy, FN_dummy)
    test_scores_TP_null = model_utils.precision_recall_f1_support(0, FP_dummy, FN_dummy)
    test_scores_FP_null = model_utils.precision_recall_f1_support(TP_dummy, 0, FN_dummy)
    f1_FP_null = 2 * 1. * rec_dummy / (1. + rec_dummy)
    test_scores_FN_null = model_utils.precision_recall_f1_support(TP_dummy, FP_dummy, 0)
    f1_FN_null = 2 * prec_dummy * 1. / (prec_dummy + 1.)
    test_scores_all_null = model_utils.precision_recall_f1_support(0, 0, 0)

    assert test_scores_no_null == (prec_dummy, rec_dummy, f1_dummy, support_dummy)
    assert test_scores_TP_null == (0., 0., 0., FN_dummy)
    assert test_scores_FP_null == (1., rec_dummy, f1_FP_null, support_dummy)
    assert test_scores_FN_null == (prec_dummy, 1., f1_FN_null, TP_dummy)
    assert test_scores_all_null == (0., 0., 0., 0)

def test_get_train_valid_indices(multi_task_lstm_crf_single_model, train_valid_indices_single_model):
    """Asserts that indices returned by the _get_train_valid_indices() of
    a MutliTaskLSTMCRf object are as expected."""
    # len of outer list
    assert len(train_valid_indices_single_model) == len(multi_task_lstm_crf_single_model.ds)
    # len of inner list
    assert len(train_valid_indices_single_model[0]) == multi_task_lstm_crf_single_model.config.k_folds
    # len of inner tuples
    assert len(train_valid_indices_single_model[0][0]) == 2

def test_get_data_partitions(multi_task_lstm_crf_single_model, data_partitions_single_model):
    """Asserts that partitions returned by the get_data_partitions() of
    a MutliTaskLSTMCRf object are as expected."""
    assert len(data_partitions_single_model) == len(multi_task_lstm_crf_single_model.ds)
    assert len(data_partitions_single_model[0]) == 6

def test_get_metrics(multi_task_lstm_crf_single_model, metrics_single_model):
    """Asserts that list of Metrics objects returned by get_metrics() is as expected."""
    ds_ = multi_task_lstm_crf_single_model.ds

    assert all(isinstance(m, Metrics) for m in metrics_single_model)
    assert isinstance(metrics_single_model, list)
    assert len(metrics_single_model) == len(ds_)
