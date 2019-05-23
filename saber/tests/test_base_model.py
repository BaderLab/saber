"""Any and all unit tests for the BaseKerasModel (saber/models/base_model.py).
"""
from ..models.base_model import BaseModel
from ..models.base_model import BaseKerasModel
from ..models.base_model import BasePyTorchModel
from .resources.constants import *
from .. import constants


def test_attributes_init_of_single_base_model(dummy_config, dummy_dataset_1, single_base_model):
    """Asserts instance attributes are initialized correctly when single `BaseModel` model is
    initialized.
    """
    assert isinstance(single_base_model, BaseModel)

    # attributes that are passed to __init__
    assert single_base_model.config is dummy_config
    assert single_base_model.datasets[0] is dummy_dataset_1
    assert single_base_model.embeddings is None

    # other instance attributes
    assert single_base_model.model is None

    # test that we can pass arbitrary keyword arguments
    assert single_base_model.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_compound_base_model(dummy_config,
                                                dummy_dataset_1,
                                                dummy_dataset_2,
                                                compound_base_model):
    """Asserts instance attributes are initialized correctly when compound `BaseModel` model is
    initialized.
    """
    assert isinstance(compound_base_model, BaseModel)

    # attributes that are passed to __init__
    assert compound_base_model.config is dummy_config
    assert compound_base_model.datasets[0] is dummy_dataset_1
    assert compound_base_model.datasets[1] is dummy_dataset_2
    assert compound_base_model.embeddings is None

    # other instance attributes
    assert compound_base_model.model is None

    # test that we can pass arbitrary keyword arguments
    assert compound_base_model.totally_arbitrary == 'arbitrary'


def test_reset_model(single_base_model):
    pass


def test_attributes_init_of_single_base_keras_model(dummy_config,
                                                    dummy_dataset_1,
                                                    single_base_keras_model):
    """Asserts instance attributes are initialized correctly when single `BaseKerasModel` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(single_base_keras_model, (BaseModel, BaseKerasModel))

    # attributes that are passed to __init__
    assert single_base_keras_model.config is dummy_config
    assert single_base_keras_model.datasets[0] is dummy_dataset_1
    assert single_base_keras_model.embeddings is None

    # other instance attributes
    assert single_base_keras_model.model is None
    assert single_base_keras_model.framework == constants.KERAS

    # test that we can pass arbitrary keyword arguments
    assert single_base_keras_model.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_compound_base_keras_model(dummy_config,
                                                      dummy_dataset_1,
                                                      dummy_dataset_2,
                                                      compound_base_keras_model):
    """Asserts instance attributes are initialized correctly when compound `BaseKerasModel` model is
    initialized without embeddings (`embeddings` attribute is None.)
    """
    assert isinstance(compound_base_keras_model, (BaseModel, BaseKerasModel))

    # attributes that are passed to __init__
    assert compound_base_keras_model.config is dummy_config
    assert compound_base_keras_model.datasets[0] is dummy_dataset_1
    assert compound_base_keras_model.datasets[1] is dummy_dataset_2
    assert compound_base_keras_model.embeddings is None

    # other instance attributes
    assert compound_base_keras_model.model is None
    assert compound_base_keras_model.framework == constants.KERAS

    # test that we can pass arbitrary keyword arguments
    assert compound_base_keras_model.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_single_model_embeddings(dummy_config,
                                                    dummy_dataset_1,
                                                    dummy_embeddings,
                                                    single_base_keras_model_embeddings):
    """Asserts instance attributes are initialized correctly when single `BaseKerasModel` model is
    initialized with embeddings (`embeddings` attribute is not None.)
    """
    assert isinstance(single_base_keras_model_embeddings, (BaseModel, BaseKerasModel))

    # attributes that are passed to __init__
    assert single_base_keras_model_embeddings.config is dummy_config
    assert single_base_keras_model_embeddings.datasets[0] is dummy_dataset_1
    assert single_base_keras_model_embeddings.embeddings is dummy_embeddings

    # other instance attributes
    assert single_base_keras_model_embeddings.model is None
    assert single_base_keras_model_embeddings.framework == constants.KERAS

    # test that we can pass arbitrary keyword arguments
    assert single_base_keras_model_embeddings.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_single_base_pytorch_model(dummy_config,
                                                      dummy_dataset_1,
                                                      single_base_pytorch_model):
    """Asserts instance attributes are initialized correctly when single `BasePyTorchModel` model is
    initialized.
    """
    assert isinstance(single_base_pytorch_model, (BaseModel, BasePyTorchModel))

    # attributes that are passed to __init__
    assert single_base_pytorch_model.config is dummy_config
    assert single_base_pytorch_model.datasets[0] is dummy_dataset_1
    assert single_base_pytorch_model.embeddings is None

    # other instance attributes
    assert single_base_pytorch_model.model is None
    assert single_base_pytorch_model.framework == constants.PYTORCH

    # test that we can pass arbitrary keyword arguments
    assert single_base_pytorch_model.totally_arbitrary == 'arbitrary'


def test_attributes_init_of_compound_base_pytorch_model(dummy_config,
                                                        dummy_dataset_1,
                                                        dummy_dataset_2,
                                                        compound_base_pytorch_model):
    """Asserts instance attributes are initialized correctly when single `BasePyTorchModel` model is
    initialized.
    """
    assert isinstance(compound_base_pytorch_model, (BaseModel, BasePyTorchModel))

    # attributes that are passed to __init__
    assert compound_base_pytorch_model.config is dummy_config
    assert compound_base_pytorch_model.datasets[0] is dummy_dataset_1
    assert compound_base_pytorch_model.datasets[1] is dummy_dataset_2
    assert compound_base_pytorch_model.embeddings is None

    # other instance attributes
    assert compound_base_pytorch_model.model is None
    assert compound_base_pytorch_model.framework == constants.PYTORCH

    # test that we can pass arbitrary keyword arguments
    assert compound_base_pytorch_model.totally_arbitrary == 'arbitrary'
