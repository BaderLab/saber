import os
import pytest

from .. import constants
from ..utils import model_utils
from ..config import Config
from ..metrics import Metrics
from ..sequence_processor import SequenceProcessor

PATH_TO_DUMMY_CONFIG = os.path.abspath('saber/tests/resources/dummy_config.ini')
PATH_TO_DUMMY_DATASET = os.path.abspath('saber/tests/resources/dummy_dataset_1')
PATH_TO_METRICS_OUTPUT = 'totally/arbitrary'

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object after parsing the dummy
    config file."""
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
    seq_processor_with_single_ds = SequenceProcessor(dummy_config)
    seq_processor_with_single_ds.load_dataset()
    seq_processor_with_single_ds.load_embeddings()
    seq_processor_with_single_ds.create_model()
    multi_task_lstm_crf_single_model = seq_processor_with_single_ds.model

    return multi_task_lstm_crf_single_model

@pytest.fixture
def dummy_output_directory(dummy_config):
    """Returns a list of ouput directories, one for each dataset."""
    return model_utils.prepare_output_directory(dummy_config.dataset_folder,
                                                dummy_config.output_folder)

@pytest.fixture
def dummy_metrics_single_model(multi_task_lstm_crf_single_model, dummy_output_directory):
    """Returns a Metrics object for the given model, 'multi_task_lstm_crf_single_model'"""
    model = multi_task_lstm_crf_single_model
    ds = model.ds[0]

    training_data = {
        'X_train': [ds.idx_seq['train']['word'], ds.idx_seq['train']['char']],
        'X_valid': None,
        'X_test': None,
        'y_train': ds.idx_seq['train']['tag'],
        'y_valid': None,
        'y_test': None,
    }

    return Metrics(training_data=training_data, idx_to_tag=ds.idx_to_tag, output_dir=dummy_output_directory[0])

def test_attributes_after_initilization_of_metrics(multi_task_lstm_crf_single_model,
                                                   dummy_metrics_single_model,
                                                   dummy_output_directory):
    """Asserts instance attributes are initialized correctly when Metrics object is initialized."""
    model = multi_task_lstm_crf_single_model
    ds = model.ds[0]
    metrics = dummy_metrics_single_model
    output_dir = dummy_output_directory[0]

    X_train_word, X_train_char = metrics.training_data['X_train']
    y_train = metrics.training_data['y_train']

    assert X_train_word.shape == ds.idx_seq['train']['word'].shape
    assert X_train_char.shape == ds.idx_seq['train']['char'].shape
    assert y_train.shape == ds.idx_seq['train']['tag'].shape

    assert metrics.idx_to_tag == ds.idx_to_tag
    assert metrics.output_dir == output_dir
    assert metrics.criteria == 'exact'
    assert metrics.current_epoch == 0
    assert metrics.current_fold == None
    assert metrics.performance_metrics_per_epoch == {p: [] for p in constants.PARTITIONS}


def test_precision_recall_f1_support_errors():
    """Asserts that call to Metrics.get_precision_recall_f1_support raises a ValueError error when
    an invalid value for parameter 'criteria' is passed."""
    # these are totally arbitrary
    y_true = [('test', 0, 3), ('test', 4, 7), ('test', 8, 11)]
    y_pred = [('test', 0, 3), ('test', 4, 7), ('test', 8, 11)]

    # anything but 'exact', 'left', or 'right' should throw an error
    invalid_args = ['right ', 'LEFT', 'eXact', 0, []]

    for arg in invalid_args:
        with pytest.raises(ValueError):
            Metrics.get_precision_recall_f1_support(y_true, y_pred, criteria=arg)
