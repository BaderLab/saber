import pytest
from torch import nn

from .. import constants
from ..models.base_model import BaseKerasModel
from ..models.base_model import BaseModel
from ..models.base_model import BasePyTorchModel


class TestBaseModel(object):
    """Collects all unit tests for `saber.models.base_model.BaseModel`.
    """
    def test_attributes_init_of_base_model(self,
                                           dummy_config,
                                           conll2003datasetreader_load,
                                           base_model):
        """Asserts instance attributes are initialized correctly when single `BaseModel` model is
        initialized.
        """
        assert isinstance(base_model, BaseModel)

        # attributes that are passed to __init__
        assert base_model.config is dummy_config
        assert base_model.datasets[0] is conll2003datasetreader_load
        assert base_model.embeddings is None

        # other instance attributes
        assert base_model.model is None

        # test that we can pass arbitrary keyword arguments
        assert base_model.totally_arbitrary == 'arbitrary'

    def test_attributes_init_of_mt_base_model(self,
                                              dummy_config_compound_dataset,
                                              conll2003datasetreader_load,
                                              dummy_dataset_2,
                                              mt_base_model):
        """Asserts instance attributes are initialized correctly when compound `BaseModel` model is
        initialized.
        """
        assert isinstance(mt_base_model, BaseModel)

        # attributes that are passed to __init__
        assert mt_base_model.config is dummy_config_compound_dataset
        assert mt_base_model.datasets[0] is conll2003datasetreader_load
        assert mt_base_model.datasets[1] is dummy_dataset_2
        assert mt_base_model.embeddings is None

        # other instance attributes
        assert mt_base_model.model is None

        # test that we can pass arbitrary keyword arguments
        assert mt_base_model.totally_arbitrary == 'arbitrary'

    def test_reset_model(self, bilstm_crf_model_specify):
        """Asserts that a new model object was created after call to `reset_model()`.
        """
        before_reset = bilstm_crf_model_specify.model

        bilstm_crf_model_specify.reset_model()
        after_reset = bilstm_crf_model_specify.model

        assert before_reset is not after_reset


class TestBaseKerasModel(object):
    """Collects all unit tests for `saber.models.base_model.BaseKerasModel`.
    """
    def test_attributes_init_of_base_keras_model(self,
                                                 dummy_config,
                                                 conll2003datasetreader_load,
                                                 base_keras_model):
        """Asserts instance attributes are initialized correctly when single `BaseKerasModel` model is
        initialized without embeddings (`embeddings` attribute is None.)
        """
        assert isinstance(base_keras_model, (BaseModel, BaseKerasModel))

        # attributes that are passed to __init__
        assert base_keras_model.config is dummy_config
        assert base_keras_model.datasets[0] is conll2003datasetreader_load
        assert base_keras_model.embeddings is None

        # other instance attributes
        assert base_keras_model.model is None
        assert base_keras_model.framework == constants.KERAS

        # test that we can pass arbitrary keyword arguments
        assert base_keras_model.totally_arbitrary == 'arbitrary'

    def test_attributes_init_of_mt_base_keras_model(self,
                                                    dummy_config_compound_dataset,
                                                    conll2003datasetreader_load,
                                                    dummy_dataset_2,
                                                    mt_base_keras_model):
        """Asserts instance attributes are initialized correctly when compound `BaseKerasModel` model is
        initialized without embeddings (`embeddings` attribute is None.)
        """
        assert isinstance(mt_base_keras_model, (BaseModel, BaseKerasModel))

        # attributes that are passed to __init__
        assert mt_base_keras_model.config is dummy_config_compound_dataset
        assert mt_base_keras_model.datasets[0] is conll2003datasetreader_load
        assert mt_base_keras_model.datasets[1] is dummy_dataset_2
        assert mt_base_keras_model.embeddings is None

        # other instance attributes
        assert mt_base_keras_model.model is None
        assert mt_base_keras_model.framework == constants.KERAS

        # test that we can pass arbitrary keyword arguments
        assert mt_base_keras_model.totally_arbitrary == 'arbitrary'

    def test_attributes_init_of_single_model_embeddings(self,
                                                        dummy_config,
                                                        conll2003datasetreader_load,
                                                        dummy_embeddings,
                                                        base_keras_model_embeddings):
        """Asserts instance attributes are initialized correctly when single `BaseKerasModel` model is
        initialized with embeddings (`embeddings` attribute is not None.)
        """
        assert isinstance(base_keras_model_embeddings, (BaseModel, BaseKerasModel))

        # attributes that are passed to __init__
        assert base_keras_model_embeddings.config is dummy_config
        assert base_keras_model_embeddings.datasets[0] is conll2003datasetreader_load
        assert base_keras_model_embeddings.embeddings is dummy_embeddings

        # other instance attributes
        assert base_keras_model_embeddings.model is None
        assert base_keras_model_embeddings.framework == constants.KERAS

        # test that we can pass arbitrary keyword arguments
        assert base_keras_model_embeddings.totally_arbitrary == 'arbitrary'

    def test_prune_output_layers_value_error(self, bilstm_crf_model_specify):
        """Asserts that `BaseKerasModel.prune_output_layers()` returns a ValueError when it is
        called from a model with only one output layer."""
        with pytest.raises(ValueError):
            bilstm_crf_model_specify.prune_output_layers([0])

    def test_prune_output_layers(self, mt_bilstm_crf_model_specify):
        """Asserts that the expected output layers are retained after a call to
        `BaseKerasModel.prune_output_layers()`.
        """
        expected = mt_bilstm_crf_model_specify.model.get_layer(index=-2)

        mt_bilstm_crf_model_specify.prune_output_layers([-2])
        actual = mt_bilstm_crf_model_specify.model.get_layer(index=-1)

        assert expected == actual


class TestBasePyTorchModel(object):
    """Collects all unit tests for `saber.models.base_model.BasePyTorchModel`.
    """
    def test_attributes_init_of_base_pytorch_model(self,
                                                   dummy_config,
                                                   conll2003datasetreader_load,
                                                   base_pytorch_model):
        """Asserts instance attributes are initialized correctly when single `BasePyTorchModel` model is
        initialized.
        """
        assert isinstance(base_pytorch_model, (BaseModel, BasePyTorchModel))

        # attributes that are passed to __init__
        assert base_pytorch_model.config is dummy_config
        assert base_pytorch_model.datasets[0] is conll2003datasetreader_load
        assert base_pytorch_model.embeddings is None

        # other instance attributes
        assert base_pytorch_model.model is None
        assert base_pytorch_model.framework == constants.PYTORCH

        # test that we can pass arbitrary keyword arguments
        assert base_pytorch_model.totally_arbitrary == 'arbitrary'

    def test_attributes_init_of_mt_base_pytorch_model(self,
                                                      dummy_config_compound_dataset,
                                                      conll2003datasetreader_load,
                                                      dummy_dataset_2,
                                                      mt_base_pytorch_model):
        """Asserts instance attributes are initialized correctly when single `BasePyTorchModel` model is
        initialized.
        """
        assert isinstance(mt_base_pytorch_model, (BaseModel, BasePyTorchModel))

        # attributes that are passed to __init__
        assert mt_base_pytorch_model.config is dummy_config_compound_dataset
        assert mt_base_pytorch_model.datasets[0] is conll2003datasetreader_load
        assert mt_base_pytorch_model.datasets[1] is dummy_dataset_2
        assert mt_base_pytorch_model.embeddings is None

        # Other instance attributes
        assert mt_base_pytorch_model.model is None
        assert mt_base_pytorch_model.framework == constants.PYTORCH

        # Test that we can pass arbitrary keyword arguments
        assert mt_base_pytorch_model.totally_arbitrary == 'arbitrary'

    def test_prune_output_layers_value_error(self, bert_for_ner_model_specify):
        """Asserts that `BaseKerasModel.prune_output_layers()` returns a ValueError when it is
        called from a model with only one output layer."""
        with pytest.raises(ValueError):
            bert_for_ner_model_specify.prune_output_layers([0])

    def test_prune_output_layers(self, mt_bert_for_ner_model_specify):
        """Asserts that the expected output layers are retained after a call to
        `BasePyTorchModel.prune_output_layers()`.
        """
        expected = nn.ModuleList([mt_bert_for_ner_model_specify.model.classifier[-1]])
        actual = mt_bert_for_ner_model_specify.prune_output_layers([-1]).classifier

        # Comparing ModuleList to ModuleList with == wasn't working, so I wrote these
        # three tests to achieve the same thing.
        assert isinstance(actual, nn.ModuleList)
        assert len(expected) == len(actual)
        assert expected[0] == actual[0]
