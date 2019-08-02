"""Test suite for the `BaseModel` class (saber.models.base_model.BaseModel).
"""
import pytest
from torch import nn

from ..models.base_model import BaseModel


class TestBaseModel(object):
    """Collects all unit tests for `saber.models.base_model.BaseModel`.
    """
    def test_initialization(self, dummy_config, conll2003datasetreader_load, base_model):
        """Asserts instance attributes are initialized correctly when single `BaseModel` model is
        initialized.
        """
        assert isinstance(base_model, BaseModel)

        # Attributes that are passed to __init__
        assert base_model.config is dummy_config
        assert base_model.datasets[0] is conll2003datasetreader_load
        assert base_model.embeddings is None

        # Other instance attributes
        assert base_model.model is None

    def test_initialization_mt(self,
                               dummy_config_compound_dataset,
                               conll2003datasetreader_load,
                               dummy_dataset_2,
                               base_model_mt):
        """Asserts instance attributes are initialized correctly when compound `BaseModel` model is
        initialized.
        """
        assert isinstance(base_model_mt, BaseModel)

        # Attributes that are passed to __init__
        assert base_model_mt.config is dummy_config_compound_dataset
        assert base_model_mt.datasets[0] is conll2003datasetreader_load
        assert base_model_mt.datasets[1] is dummy_dataset_2
        assert base_model_mt.embeddings is None

        # Other instance attributes
        assert base_model_mt.model is None

    def test_reset_model(self, bert_for_ner_specify):
        """Asserts that a new model object was created after call to `reset_model()`.
        """
        before_reset = bert_for_ner_specify.model

        bert_for_ner_specify.reset_model()
        after_reset = bert_for_ner_specify.model

        assert before_reset is not after_reset

    def test_reset_model_mt(self, bert_for_ner_specify_mt):
        """Asserts that a new model object was created after call to `reset_model()`.
        """
        before_reset = bert_for_ner_specify_mt.model

        bert_for_ner_specify_mt.reset_model()
        after_reset = bert_for_ner_specify_mt.model

        assert before_reset is not after_reset

    def test_prune_output_layers_value_error(self, bert_for_ner_specify):
        """Asserts that `BaseModel.prune_output_layers()` returns a ValueError when it is
        called from a model with only one output layer."""
        with pytest.raises(ValueError):
            bert_for_ner_specify.prune_output_layers([0])

    def test_prune_output_layers(self, bert_for_ner_specify_mt):
        """Asserts that the expected output layers are retained after a call to
        `BaseModel.prune_output_layers()`.
        """
        expected = nn.ModuleList([bert_for_ner_specify_mt.model.classifier[-1]])
        actual = bert_for_ner_specify_mt.prune_output_layers([-1]).classifier

        # Comparing ModuleList to ModuleList with == wasn't working, so I wrote these three tests to
        # achieve the same thing.
        assert isinstance(actual, nn.ModuleList)
        assert len(expected) == len(actual)
        assert expected[0] == actual[0]
