import os

import pytest
import torch
from keras.callbacks import ModelCheckpoint, TensorBoard

from .. import constants
from ..utils import model_utils
from .resources.constants import *
import numpy as np


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

    def test_setup_checkpoint_callback(self, dummy_config, dummy_output_dir):
        """Check that we get the expected results from call to
        `model_utils.setup_checkpoint_callback()`.
        """
        simple_actual = model_utils.setup_checkpoint_callback(dummy_config, dummy_output_dir)
        blank_actual = model_utils.setup_checkpoint_callback(dummy_config, [])

        # Should get as many Callback objects as datasets
        assert len(dummy_output_dir) == len(simple_actual)
        # All objects in returned list should be of type ModelCheckpoint
        assert all([isinstance(x, ModelCheckpoint) for x in simple_actual])

        # Blank input should return blank list
        assert blank_actual == []

    def test_setup_tensorboard_callback(self, dummy_output_dir):
        """Check that we get the expected results from call to
        `model_utils.setup_tensorboard_callback()`.
        """
        simple_actual = model_utils.setup_tensorboard_callback(dummy_output_dir)
        blank_actual = model_utils.setup_tensorboard_callback([])

        # Should get as many Callback objects as datasets
        assert len(dummy_output_dir) == len(simple_actual)
        # All objects in returned list should be of type TensorBoard
        assert all([isinstance(x, TensorBoard) for x in simple_actual])

        # Blank input should return blank list
        assert blank_actual == []

    def test_setup_metrics_callback(self):
        """
        """
        pass

    def test_setup_callbacks(self, dummy_config, dummy_output_dir):
        """Check that we get the expected results from call to
        `model_utils.setup_callbacks()`.
        """
        # Setup callbacks with config.tensorboard == True
        dummy_config.tensorboard = True
        with_tensorboard_actual = model_utils.setup_callbacks(dummy_config, dummy_output_dir)
        # Setup callbacks with config.tensorboard == False
        dummy_config.tensorboard = False
        without_tensorboard_actual = model_utils.setup_callbacks(dummy_config, dummy_output_dir)

        blank_actual = []

        # Should get as many Callback objects as datasets
        assert all([len(x) == len(dummy_output_dir) for x in with_tensorboard_actual])
        assert all([len(x) == len(dummy_output_dir) for x in without_tensorboard_actual])

        # All objects in returned list should be of expected type
        assert all([isinstance(x, ModelCheckpoint) for x in with_tensorboard_actual[0]])
        assert all([isinstance(x, TensorBoard) for x in with_tensorboard_actual[1]])
        assert all([isinstance(x, ModelCheckpoint) for x in without_tensorboard_actual[0]])

        # Blank input should return blank list
        assert blank_actual == []

    def test_precision_recall_f1_support(self):
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

    def test_get_keras_optimizer_value_error(self):
        """Asserts that `model_utils.get_keras_optimizer()` returns a ValueError when an invalid
        argument for `optimizer` is passed.
        """
        with pytest.raises(ValueError):
            model_utils.get_keras_optimizer('invalid')

    def test_mask_labels(self):
        """Assert that `model_utils.mask_pads()` returns the expected values.
        """
        y_true = np.concatenate((np.ones([10, 90]), np.zeros([10, 10])), axis=-1)
        y_pred = np.concatenate((np.ones([10, 80]), np.zeros([10, 20])), axis=-1)

        expected = (y_true[y_true == 1].reshape(10, 90),
                    y_pred[:, 0:90])

        actual = model_utils.mask_labels(y_true, y_pred, constants.PAD_VALUE)

        for exp, act in zip(expected, actual):
            assert exp.tolist() == act

    def test_freeze_output_layers(self, saber_compound_dataset_model):
        """Asserts that model_utils.freeze_output_layers() freezes the expected layers.
        """
        # Get model, set first output as the output layer currently being trained
        model = saber_compound_dataset_model.models[0].model
        model_idx = 0

        # Check that all layers are trainble before calling freeze_output_layers
        assert all(model.get_layer(f'crf_{i}').trainable for i, _ in enumerate(model.output))

        # Freeze all output layers but the 0-th
        model_utils.freeze_output_layers(model, model_idx=0)

        assert model.get_layer(f'crf_{model_idx}').trainable
        assert not model.get_layer(f'crf_{1}').trainable

    def test_get_targets(self, mt_bilstm_crf_model):
        """Asserts that `model_utils.get_targets()` returns the expected values for a simple example.
        """
        # Get training data, set first output as the output layer currently being trained
        training_data = mt_bilstm_crf_model.prepare_data_for_training()
        model_idx = 0

        expected = (
            [training_data[model_idx]['y_train'], np.zeros_like(training_data[model_idx]['y_train'])],
            [training_data[model_idx]['y_valid'], np.zeros_like(training_data[model_idx]['y_valid'])]
        )

        actual = model_utils.get_targets(training_data, model_idx=0)

        for i, _ in enumerate(expected):
            for j, _ in enumerate(expected):
                assert np.all(np.equal(expected[i][j], actual[i][j]))

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
