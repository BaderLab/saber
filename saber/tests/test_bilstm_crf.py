"""Any and all unit tests for the BiLSTMCRF (saber/models/bilstm_crf.py).
"""
from keras.engine.training import Model

from ..models.base_model import BaseKerasModel
from ..models.base_model import BaseModel
from ..models.bilstm_crf import BiLSTMCRF
from .resources.constants import *
from keras_contrib.layers.crf import CRF
from keras.optimizers import Optimizer
import numpy as np


def test_attributes_init_of_single_mtbilstm_model(dummy_config, dummy_dataset_1, bilstm_crf_model):
    """Asserts instance attributes are initialized correctly when single `BiLSTMCRF` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(bilstm_crf_model, (BiLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert bilstm_crf_model.config is dummy_config
    assert bilstm_crf_model.datasets[0] is dummy_dataset_1
    assert bilstm_crf_model.embeddings is None
    # other instance attributes
    assert bilstm_crf_model.model is None
    # test that we can pass arbitrary keyword arguments
    assert bilstm_crf_model.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_bilstm_crf_model_specify(dummy_config,
                                                     dummy_dataset_1,
                                                     bilstm_crf_model_specify):
    """Asserts instance attributes are initialized correctly when single `BiLSTMCRF`
    model is initialized without embeddings (`embeddings` attribute is None) and
    `BiLSTMCRF.specify()` has been called.
    """
    assert isinstance(bilstm_crf_model_specify, (BiLSTMCRF, BaseKerasModel))

    # attributes that are passed to __init__
    assert bilstm_crf_model_specify.config is dummy_config
    assert bilstm_crf_model_specify.datasets[0] is dummy_dataset_1
    assert bilstm_crf_model_specify.embeddings is None

    # other instance attributes
    assert isinstance(bilstm_crf_model_specify.model, Model)

    # test that we can pass arbitrary keyword arguments
    assert bilstm_crf_model_specify.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_bilstm_crf_model_embeddings(dummy_config,
                                                        dummy_dataset_1,
                                                        dummy_embeddings,
                                                        bilstm_crf_model_embeddings):
    """Asserts instance attributes are initialized correctly when single `BiLSTMCRF` model is
    initialized with embeddings (`embeddings` attribute is not None.)
    """
    assert isinstance(bilstm_crf_model_embeddings, (BiLSTMCRF, BaseKerasModel, BaseModel))
    # attributes that are passed to __init__
    assert bilstm_crf_model_embeddings.config is dummy_config
    assert bilstm_crf_model_embeddings.datasets[0] is dummy_dataset_1
    assert bilstm_crf_model_embeddings.embeddings is dummy_embeddings
    # other instance attributes
    assert bilstm_crf_model_embeddings.model is None
    # test that we can pass arbitrary keyword arguments
    assert bilstm_crf_model_embeddings.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_bilstm_crf_model_embeddings_specify(dummy_config,
                                                                dummy_dataset_1,
                                                                dummy_embeddings,
                                                                bilstm_crf_model_embeddings_specify):
    """Asserts instance attributes are initialized correctly when single BiLSTMCRF
    model is initialized with embeddings (`embeddings` attribute is not None) and
    `BiLSTMCRF.specify()` has been called.
    """
    assert isinstance(bilstm_crf_model_embeddings_specify, (BiLSTMCRF, BaseKerasModel))
    # attributes that are passed to __init__
    assert bilstm_crf_model_embeddings_specify.config is dummy_config
    assert bilstm_crf_model_embeddings_specify.datasets[0] is dummy_dataset_1
    assert bilstm_crf_model_embeddings_specify.embeddings is dummy_embeddings
    # other instance attributes
    assert isinstance(bilstm_crf_model_embeddings_specify.model, Model)
    # test that we can pass arbitrary keyword arguments
    assert bilstm_crf_model_embeddings_specify.totally_arbitrary == 'arbitrary'


def test_prepare_data_for_training(dummy_dataset_1, bilstm_crf_model):
    """Assert that the values returned from call to `BaseKerasModel.prepare_data_for_training()` are
    as expected.
    """
    training_data = bilstm_crf_model.prepare_data_for_training()
    partitions = ['x_train', 'y_train', 'x_valid', 'y_valid', 'x_test', 'y_test']

    # assert each item in training_data contains the expected keys
    assert all(partition in data for data in training_data for partition in partitions)

    # assert that the items in training_data contain the expected values
    assert all(data['x_train'] == [dummy_dataset_1.idx_seq['train']['word'],
                                   dummy_dataset_1.idx_seq['train']['char']]
               for data in training_data)
    assert all(data['x_valid'] == [dummy_dataset_1.idx_seq['valid']['word'],
                                   dummy_dataset_1.idx_seq['valid']['char']]
               for data in training_data)
    assert all(data['x_test'] == [dummy_dataset_1.idx_seq['test']['word'],
                                  dummy_dataset_1.idx_seq['test']['char']]
               for data in training_data)
    assert all(np.array_equal(data['y_train'], dummy_dataset_1.idx_seq['train']['tag'])
               for data in training_data)
    assert all(np.array_equal(data['y_valid'], dummy_dataset_1.idx_seq['valid']['tag'])
               for data in training_data)
    assert all(np.array_equal(data['y_test'], dummy_dataset_1.idx_seq['test']['tag'])
               for data in training_data)


def test_crf_after_transfer(bilstm_crf_model_specify, dummy_dataset_1, dummy_dataset_2):
    """Asserts that the CRF output layer of a model is replaced with a new layer when
    `BiLSTMCRF.prepare_for_transfer()` is called by testing that the `name` attribute
    of the final layer.
    """
    # shorten test statements
    model = bilstm_crf_model_specify

    # get output layer names before transfer
    expected_name_before_transfer = 'crf_0'
    actual_name_before_transfer = model.model.layers[-1].name

    # get output layer names after transfer
    model.prepare_for_transfer([dummy_dataset_1, dummy_dataset_2])

    expected_names_after_transfer = ['crf_0', 'crf_1']
    actual_names_after_transfer = []
    actual_outputs_after_transfer = []
    for i in range(-len(expected_names_after_transfer), 0):
        actual_names_after_transfer.append(model.model.layers[i].name)
        actual_outputs_after_transfer.append(model.model.layers[i])

    assert expected_name_before_transfer == actual_name_before_transfer
    assert expected_names_after_transfer == actual_names_after_transfer
    assert all(isinstance(output, CRF) for output in actual_outputs_after_transfer)


def test_prepare_optimizers_single_dataset(bilstm_crf_model_specify):
    """Assert that `BiLSTMCRF.prepare_optimizers()` returns a list of Keras optimizers
    for a single-task model.
    """
    actual = bilstm_crf_model_specify.prepare_optimizers()

    assert all(isinstance(opt, Optimizer) for opt in actual)


def test_prepare_optimizers_compound_dataset(mt_bilstm_crf_model_specify):
    """Assert that `BiLSTMCRF.prepare_optimizers()` returns a list of Keras optimizers
    for a multi-task model.
    """
    actual = mt_bilstm_crf_model_specify.prepare_optimizers()

    assert all(isinstance(opt, Optimizer) for opt in actual)
