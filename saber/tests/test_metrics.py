"""Any and all unit tests for the Metrics class (saber/metrics.py).
"""
import pytest

from .. import constants
from ..config import Config
from ..dataset import Dataset
from ..metrics import Metrics
from ..utils import model_utils
from .resources.dummy_constants import *

PATH_TO_METRICS_OUTPUT = 'totally/arbitrary'

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    dummy_config = Config(PATH_TO_DUMMY_CONFIG)
    return dummy_config

@pytest.fixture
def dummy_dataset():
    """Returns a single dummy Dataset instance after calling Dataset.load().
    """
    # Don't replace rare tokens for the sake of testing
    dataset = Dataset(directory=PATH_TO_DUMMY_DATASET_1, replace_rare_tokens=False)
    dataset.load()

    return dataset

@pytest.fixture
def dummy_output_dir(tmpdir, dummy_config):
    """Returns list of output directories."""
    # make sure top-level directory is the pytest tmpdir
    dummy_config.output_folder = tmpdir.strpath
    output_dirs = model_utils.prepare_output_directory(dummy_config)

    return output_dirs

@pytest.fixture
def dummy_training_data(dummy_dataset):
    """Returns training data from `dummy_dataset`.
    """
    training_data = {'x_train': [dummy_dataset.idx_seq['train']['word'],
                                 dummy_dataset.idx_seq['train']['char']],
                     'x_valid': None,
                     'x_test': None,
                     'y_train': dummy_dataset.idx_seq['train']['tag'],
                     'y_valid': None,
                     'y_test': None,
                    }

    return training_data

@pytest.fixture
def dummy_metrics(dummy_config, dummy_dataset, dummy_training_data, dummy_output_dir):
    """Returns an instance of Metrics.
    """
    metrics = Metrics(config=dummy_config,
                      training_data=dummy_training_data,
                      index_map=dummy_dataset.idx_to_tag,
                      output_dir=dummy_output_dir,
                      # to test passing of arbitrary keyword args to constructor
                      totally_arbitrary='arbitrary')
    return metrics

############################################ UNIT TESTS ############################################

def test_attributes_after_initilization(dummy_config,
                                        dummy_dataset,
                                        dummy_output_dir,
                                        dummy_training_data,
                                        dummy_metrics):
    """Asserts instance attributes are initialized correctly when Metrics object is initialized."""
    # attributes that are passed to __init__
    assert dummy_metrics.config is dummy_config
    assert dummy_metrics.training_data is dummy_training_data
    assert dummy_metrics.index_map is dummy_dataset.idx_to_tag
    assert dummy_metrics.output_dir == dummy_output_dir
    # other instance attributes
    assert dummy_metrics.current_epoch == 0
    assert dummy_metrics.performance_metrics == {p: [] for p in constants.PARTITIONS}
    # test that we can pass arbitrary keyword arguments
    assert dummy_metrics.totally_arbitrary == 'arbitrary'

def test_precision_recall_f1_support_value_error():
    """Asserts that call to `Metrics.get_precision_recall_f1_support` raises a `ValueError` error
    when an invalid value for parameter `criteria` is passed."""
    # these are totally arbitrary
    y_true = [('test', 0, 3), ('test', 4, 7), ('test', 8, 11)]
    y_pred = [('test', 0, 3), ('test', 4, 7), ('test', 8, 11)]

    # anything but 'exact', 'left', or 'right' should throw an error
    invalid_args = ['right ', 'LEFT', 'eXact', 0, []]

    for arg in invalid_args:
        with pytest.raises(ValueError):
            Metrics.get_precision_recall_f1_support(y_true, y_pred, criteria=arg)
