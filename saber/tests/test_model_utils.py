"""Test suite for the `model_utils` module (saber.utils.model_utils).
"""
import os

import torch

from .. import constants
from ..utils import model_utils


class TestModelUtils(object):
    """Collects all unit tests for `saber.utils.model_utils`.
    """
    def test_prepare_output_directory(self, dummy_config, dummy_output_dir):
        """Assert that `model_utils.prepare_output_directory()` creates the expected directories
        with the expected content.
        """
        # TODO (johngiorgi): Need to test the actual output of the function!
        # Check that the expected directories are created
        assert all([os.path.isdir(dir_) for dir_ in dummy_output_dir])
        # Check that they contain config files
        assert all([os.path.isfile(os.path.join(dir_, 'config.ini')) for dir_ in dummy_output_dir])

    def test_prepare_pretrained_model_dir(self, dummy_config):
        """Asserts that filepath returned by `generic_utils.get_pretrained_model_dir()` is as expected.
        """
        dataset = os.path.basename(dummy_config.dataset_folder[0])
        expected = os.path.join(dummy_config.output_folder, constants.PRETRAINED_MODEL_DIR, dataset)
        assert model_utils.prepare_pretrained_model_dir(dummy_config) == expected

    def test_setup_metrics_callback(self):
        """
        """
        pass

    def test_get_device_no_model(self):
        """Asserts that `model_utils.get_device()` reutnrs the expected PyTorch device and number of
        GPUs.
        """
        # The tox.ini specifices env variable CUDA_VISIBLE_DEVICES=''
        expected = (torch.device('cpu'), 0)
        actual = model_utils.get_device()

        assert expected == actual

    def test_get_device_model(self):
        """
        """
        pass
