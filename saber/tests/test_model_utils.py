"""Any and all unit tests for the model_utils (saber/utils/model_utils.py).
"""
import os

import pytest
import torch
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences

from .. import constants
from ..utils import model_utils
from .resources.constants import *
import numpy as np


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


def test_get_keras_optimizer_value_error():
    """Asserts that `model_utils.get_keras_optimizer()` returns a ValueError when an invalid
    argument for `optimizer` is passed.
    """
    with pytest.raises(ValueError):
        model_utils.get_keras_optimizer('invalid')


def test_mask_labels_no_pads():
    """Assert that `model_utils.mask_pads()` returns the expected values for a set of simple inputs.
    """
    empty_input = np.ones([100, ])

    expected = empty_input, empty_input
    actual = model_utils.mask_labels(empty_input, empty_input, constants.PAD_VALUE)

    assert np.array_equal(expected, actual)


def test_mask_labels_with_pads():
    """Assert that `model_utils.mask_pads()` returns the expected values for a set of simple inputs.
    """
    empty_input = np.concatenate((np.ones([90, ]), np.zeros([10, ])), axis=None)

    expected = empty_input[empty_input == 1], empty_input[empty_input == 1]
    actual = model_utils.mask_labels(empty_input, empty_input, constants.PAD_VALUE)

    assert np.array_equal(expected, actual)


def test_freeze_output_layers(saber_compound_dataset_model):
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


def test_get_targets(mt_bilstm_crf_model):
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


def test_get_device_no_model():
    """Asserts that `model_utils.get_device()` reutnrs the expected PyTorch device and number of
    GPUs.
    """
    # The tox.ini specifices env variable CUDA_VISIBLE_DEVICES=''
    expected = (torch.device('cpu'), 0)
    actual = model_utils.get_device()

    assert expected == actual


def test_get_device_model():
    """
    """
    pass


def test_setup_type_to_idx_for_bert(dummy_dataset_1):
    """Assert that `dummy_dataset_1.type_to_idx['tag']` is updated as expected after call to
    `model_utils.setup_type_to_idx_for_bert`.
    """
    # check that the wordpiece tag ('X') is not in type_to_idx['tag'] by default
    assert constants.WORDPIECE not in dummy_dataset_1.type_to_idx['tag']

    model_utils.setup_type_to_idx_for_bert(dummy_dataset_1)

    # check that setup_type_to_idx_for_bert has added the wordpiece tag ('X') with the correct index
    assert dummy_dataset_1.type_to_idx['tag'][constants.WORDPIECE] == \
        len(dummy_dataset_1.type_to_idx['tag']) - 1


def test_process_data_for_bert(bert_tokenizer):
    """
    """
    pass


def test_tokenize_for_bert(bert_tokenizer):
    """Asserts that the tokenized text and labels returned by `model_utils.tokenize_for_bert()` are
    as expected.
    """
    x = constants.WORDPIECE
    outside = constants.OUTSIDE

    dummy_word_seq = [['Jim', 'Henson', 'was', 'a', 'puppeteer', '.']]
    dummy_tag_seq = [['I-PER', 'I-PER', outside, outside, outside, outside]]

    expected_word_seq = [['Jim', 'He', '##nson', 'was', 'a', 'puppet', '##eer', '.']]
    expected_tag_seq = [['I-PER', 'I-PER', x, outside, outside, outside, x, outside]]

    expected = (expected_word_seq, expected_tag_seq)
    actual = model_utils.tokenize_for_bert(bert_tokenizer, dummy_word_seq, dummy_tag_seq)

    assert expected == actual


def test_type_to_idx_for_bert(bert_tokenizer):
    """Asserts that `model_utils.type_to_idx_for_bert()` returns the expected values.
    """
    x = constants.WORDPIECE
    pad = constants.PAD
    unk = constants.UNK
    outside = constants.OUTSIDE

    dummy_word_seq = [['Jim', 'He', '##nson', 'was', 'a', 'puppet', '##eer', '.']]
    dummy_tag_seq = [['I-PER', 'I-PER', x, outside, outside, outside, x, outside]]
    dummy_type_to_idx = {pad: 0, unk: 1, x: 2, 'I-PER': 3, outside: 4}

    actual = model_utils.type_to_idx_for_bert(bert_tokenizer,
                                              dummy_word_seq, dummy_tag_seq,
                                              dummy_type_to_idx)

    expected_word_indices = \
        pad_sequences([bert_tokenizer.convert_tokens_to_ids(sent) for sent in dummy_word_seq],
                      maxlen=constants.MAX_SENT_LEN,
                      dtype='long',
                      padding="post",
                      truncating="post",
                      value=constants.PAD_VALUE)

    expected_tag_indices = \
        pad_sequences([[dummy_type_to_idx.get(tag, dummy_type_to_idx[constants.UNK])
                        for tag in sent] for sent in dummy_tag_seq],
                      maxlen=constants.MAX_SENT_LEN,
                      dtype='long',
                      padding="post",
                      truncating="post",
                      value=constants.PAD_VALUE)

    expected_attention_indices = \
        [[float(idx > 0) for idx in sent] for sent in expected_word_indices]

    expected = (expected_word_indices, expected_tag_indices, expected_attention_indices)

    for exp, act in zip(expected, actual):
        assert np.allclose(exp, act)


def test_get_dataloader_for_bert():
    """
    """
    pass


def test_get_bert_optimizer(bert_tokenizer):
    """
    """
    pass
