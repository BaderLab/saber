"""Any and all unit tests for the MultiTaskLSTMCRF (saber/models/multi_task_lstm_crf.py).
"""
from keras.engine.training import Model

import pytest

from ..models.base_model import BaseKerasModel
from ..models.multi_task_lstm_crf import MultiTaskLSTMCRF
from .resources.dummy_constants import *


def test_attributes_init_of_single_mtbilstm_model(dummy_config, dummy_dataset_1, single_mtbilstm_model):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(single_mtbilstm_model, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_mtbilstm_model.config is dummy_config
    assert single_mtbilstm_model.datasets[0] is dummy_dataset_1
    assert single_mtbilstm_model.embeddings is None
    # other instance attributes
    assert single_mtbilstm_model.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_mtbilstm_model.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_mtbilstm_model_specify(dummy_config, dummy_dataset_1, single_mtbilstm_model_specify):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF`
    model is initialized without embeddings (`embeddings` attribute is None) and
    `MultiTaskLSTMCRF.specify()` has been called.
    """
    assert isinstance(single_mtbilstm_model_specify, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_mtbilstm_model_specify.config is dummy_config
    assert single_mtbilstm_model_specify.datasets[0] is dummy_dataset_1
    assert single_mtbilstm_model_specify.embeddings is None
    # other instance attributes
    assert all([isinstance(model, Model) for model in single_mtbilstm_model_specify.models])
    # test that we can pass arbitrary keyword arguments
    assert single_mtbilstm_model_specify.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_mtbilstm_model_embeddings(dummy_config, dummy_dataset_1,
                                                             dummy_embeddings,
                                                             single_mtbilstm_model_embeddings):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized with embeddings (`embeddings` attribute is not None.)
    """
    assert isinstance(single_mtbilstm_model_embeddings, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_mtbilstm_model_embeddings.config is dummy_config
    assert single_mtbilstm_model_embeddings.datasets[0] is dummy_dataset_1
    assert single_mtbilstm_model_embeddings.embeddings is dummy_embeddings
    # other instance attributes
    assert single_mtbilstm_model_embeddings.models == []
    # test that we can pass arbitrary keyword arguments
    assert single_mtbilstm_model_embeddings.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_mtbilstm_model_embeddings_specify(dummy_config, dummy_dataset_1,
                                                                     dummy_embeddings,
                                                                     single_mtbilstm_model_embeddings_specify):
    """Asserts instance attributes are initialized correctly when single MultiTaskLSTMCRF
    model is initialized with embeddings (`embeddings` attribute is not None) and
    `MultiTaskLSTMCRF.specify()` has been called.
    """
    assert isinstance(single_mtbilstm_model_embeddings_specify, (MultiTaskLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert single_mtbilstm_model_embeddings_specify.config is dummy_config
    assert single_mtbilstm_model_embeddings_specify.datasets[0] is dummy_dataset_1
    assert single_mtbilstm_model_embeddings_specify.embeddings is dummy_embeddings
    # other instance attributes
    assert all([isinstance(model, Model) for model in single_mtbilstm_model_embeddings_specify.models])
    # test that we can pass arbitrary keyword arguments
    assert single_mtbilstm_model_embeddings_specify.totally_arbitrary == 'arbitrary'

def test_prepare_data_for_training(dummy_dataset_1, single_mtbilstm_model):
    """Assert that the values returned from call to `BaseKerasModel.prepare_data_for_training()` are
    as expected.
    """
    training_data = single_mtbilstm_model.prepare_data_for_training()
    partitions = ['x_train', 'y_train', 'x_valid', 'y_valid', 'x_test', 'y_test']

    # assert each item in training_data contains the expected keys
    assert all(partition in data for data in training_data for partition in partitions)

    # assert that the items in training_data contain the expected values
    assert all(data['x_train'] == [dummy_dataset_1.idx_seq['train']['word'], dummy_dataset_1.idx_seq['train']['char']]
               for data in training_data)
    assert all(data['x_valid'] == [dummy_dataset_1.idx_seq['valid']['word'], dummy_dataset_1.idx_seq['valid']['char']]
               for data in training_data)
    assert all(data['x_test'] == [dummy_dataset_1.idx_seq['test']['word'], dummy_dataset_1.idx_seq['test']['char']]
               for data in training_data)
    assert all(np.array_equal(data['y_train'], dummy_dataset_1.idx_seq['train']['tag']) for data in training_data)
    assert all(np.array_equal(data['y_valid'], dummy_dataset_1.idx_seq['valid']['tag']) for data in training_data)
    assert all(np.array_equal(data['y_test'], dummy_dataset_1.idx_seq['test']['tag']) for data in training_data)

def test_crf_after_transfer(single_mtbilstm_model_specify, dummy_dataset_2):
    """Asserts that the CRF output layer of a model is replaced with a new layer when
    `MultiTaskLSTMCRF.prepare_for_transfer()` is called by testing that the `name` attribute
    of the final layer.
    """
    # shorten test statements
    test_model = single_mtbilstm_model_specify

    # get output layer names before transfer
    expected_before_transfer = ['crf_classifier']
    actual_before_transfer = [model.layers[-1].name for model in test_model.models]
    # get output layer names after transfer
    test_model.prepare_for_transfer([dummy_dataset_2])
    expected_after_transfer = ['target_crf_classifier']
    actual_after_transfer = [model.layers[-1].name for model in test_model.models]

    assert actual_before_transfer == expected_before_transfer
    assert actual_after_transfer == expected_after_transfer
