"""Any and all unit tests for the BaseKerasModel (saber/models/base_model.py).
"""
from ..models.base_model import BaseKerasModel
from .resources.constants import *


def test_attributes_init_of_single_model(dummy_config, dummy_dataset_1, single_base_keras_model):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(single_base_keras_model, BaseKerasModel)

    # attributes that are passed to __init__
    assert single_base_keras_model.config is dummy_config
    assert single_base_keras_model.datasets[0] is dummy_dataset_1
    assert single_base_keras_model.embeddings is None

    # other instance attributes
    assert single_base_keras_model.model is None

    # test that we can pass arbitrary keyword arguments
    assert single_base_keras_model.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_single_model_embeddings(dummy_config, dummy_dataset_1, dummy_embeddings,
                                                    single_base_keras_model_embeddings):
    """Asserts instance attributes are initialized correctly when single `MultiTaskLSTMCRF` model is
    initialized with embeddings (`embeddings` attribute is not None.)
    """
    assert isinstance(single_base_keras_model_embeddings, BaseKerasModel)

    # attributes that are passed to __init__
    assert single_base_keras_model_embeddings.config is dummy_config
    assert single_base_keras_model_embeddings.datasets[0] is dummy_dataset_1
    assert single_base_keras_model_embeddings.embeddings is dummy_embeddings

    # other instance attributes
    assert single_base_keras_model_embeddings.model is None

    # test that we can pass arbitrary keyword arguments
    assert single_base_keras_model_embeddings.totally_arbitrary == 'arbitrary'
