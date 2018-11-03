"""Any and all unit tests for the model_utils (saber/utils/model_utils.py).
"""
import os

from keras.callbacks import ModelCheckpoint, TensorBoard

import pytest

from ..config import Config
from ..utils import model_utils
from .resources.dummy_constants import *

######################################### PYTEST FIXTURES #########################################

@pytest.fixture
def dummy_config():
    """Returns an instance of a Config object."""
    return Config(PATH_TO_DUMMY_CONFIG)

@pytest.fixture
def dummy_output_dir(tmpdir, dummy_config):
    """Returns list of output directories."""
    # make sure top-level directory is the pytest tmpdir
    dummy_config.output_folder = tmpdir.strpath
    output_dirs = model_utils.prepare_output_directory(dummy_config)

    return output_dirs

############################################ UNIT TESTS ############################################

def test_prepare_output_directory(dummy_config, dummy_output_dir):
    """Assert that `model_utils.prepare_output_directory()` creates the expected directories
    with the expected content.
    """
    # TODO (johngiorgi): need to test the actual output of the function!
    # check that the expected directories are created
    assert all([os.path.isdir(dir_) for dir_ in dummy_output_dir])
    # check that they contain config files
    assert all([os.path.isfile(os.path.join(dir_, 'config.ini')) for dir_ in dummy_output_dir])

def test_prepare_pretrained_model_dir(dummy_config):
    """Asserts that filepath returned by `generic_utils.get_pretrained_model_dir()` is as expected.
    """
    dataset = os.path.basename(dummy_config.dataset_folder[0])
    expected = os.path.join(dummy_config.output_folder, constants.PRETRAINED_MODEL_DIR, dataset)
    assert model_utils.prepare_pretrained_model_dir(dummy_config) == expected

def test_setup_checkpoint_callback(dummy_config, dummy_output_dir):
    """Check that we get the expected results from call to
    `model_utils.setup_checkpoint_callback()`.
    """
    simple_actual = model_utils.setup_checkpoint_callback(dummy_config, dummy_output_dir)
    blank_actual = model_utils.setup_checkpoint_callback(dummy_config, [])

    # should get as many Callback objects as datasets
    assert len(dummy_output_dir) == len(simple_actual)
    # all objects in returned list should be of type ModelCheckpoint
    assert all([isinstance(x, ModelCheckpoint) for x in simple_actual])

    # blank input should return blank list
    assert blank_actual == []

def test_setup_tensorboard_callback(dummy_output_dir):
    """Check that we get the expected results from call to
    `model_utils.setup_tensorboard_callback()`.
    """
    simple_actual = model_utils.setup_tensorboard_callback(dummy_output_dir)
    blank_actual = model_utils.setup_tensorboard_callback([])

    # should get as many Callback objects as datasets
    assert len(dummy_output_dir) == len(simple_actual)
    # all objects in returned list should be of type TensorBoard
    assert all([isinstance(x, TensorBoard) for x in simple_actual])

    # blank input should return blank list
    assert blank_actual == []

def test_setup_metrics_callback():
    """
    """
    pass

def test_setup_callbacks(dummy_config, dummy_output_dir):
    """Check that we get the expected results from call to
    `model_utils.setup_callbacks()`.
    """
    # setup callbacks with config.tensorboard == True
    dummy_config.tensorboard = True
    with_tensorboard_actual = model_utils.setup_callbacks(dummy_config, dummy_output_dir)
    # setup callbacks with config.tensorboard == False
    dummy_config.tensorboard = False
    without_tensorboard_actual = model_utils.setup_callbacks(dummy_config, dummy_output_dir)

    blank_actual = []

    # should get as many Callback objects as datasets
    assert all([len(x) == len(dummy_output_dir) for x in with_tensorboard_actual])
    assert all([len(x) == len(dummy_output_dir) for x in without_tensorboard_actual])

    # all objects in returned list should be of expected type
    assert all([isinstance(x, ModelCheckpoint) for x in with_tensorboard_actual[0]])
    assert all([isinstance(x, TensorBoard) for x in with_tensorboard_actual[1]])
    assert all([isinstance(x, ModelCheckpoint) for x in without_tensorboard_actual[0]])

    # blank input should return blank list
    assert blank_actual == []

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
