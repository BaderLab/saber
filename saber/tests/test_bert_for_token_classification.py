"""Any and all unit tests for the BertForTokenClassification (saber/models/BertForTokenClassification.py).
"""
import pytest
from keras.engine.training import Model
from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer

from ..models.base_model import BaseModel, BasePyTorchModel
from ..models.bert_token_classifier import BertTokenClassifier
from .resources.dummy_constants import *


def test_attributes_init_of_single_model(dummy_config, dummy_dataset_1,
                                         single_bert_token_classifier_model):
    """Asserts instance attributes are initialized correctly when single `BertTokenClassifier` model
    is initialized.
    """
    assert isinstance(single_bert_token_classifier_model,
                      (BertTokenClassifier, BasePyTorchModel, BaseModel))
    # attributes that are passed to __init__
    assert single_bert_token_classifier_model.config is dummy_config
    assert single_bert_token_classifier_model.datasets[0] is dummy_dataset_1
    # other instance attributes
    assert single_bert_token_classifier_model.models == []
    assert single_bert_token_classifier_model.embeddings is None
    assert single_bert_token_classifier_model.device.type == 'cpu'
    assert single_bert_token_classifier_model.pretrained_model_name_or_path == \
        'bert-base-uncased'
    assert isinstance(single_bert_token_classifier_model.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_bert_token_classifier_model.totally_arbitrary == 'arbitrary'

def test_attributes_init_of_single_model_specify(dummy_config, dummy_dataset_1,
                                                 single_bert_token_classifier_model_specify):
    """Asserts instance attributes are initialized correctly when single `BertTokenClassifier`
    model is initialized without embeddings (`embeddings` attribute is None) and
    `BertTokenClassifier.specify()` has been called.
    """
    assert isinstance(single_bert_token_classifier_model_specify,
                      (BertTokenClassifier, BasePyTorchModel, BaseModel))
    # attributes that are passed to __init__
    assert single_bert_token_classifier_model_specify.config is dummy_config
    assert single_bert_token_classifier_model_specify.datasets[0] is dummy_dataset_1
    # other instance attributes
    assert all([isinstance(model, BertForTokenClassification)
                for model in single_bert_token_classifier_model_specify.models])
    assert single_bert_token_classifier_model_specify.embeddings is None
    assert single_bert_token_classifier_model_specify.pretrained_model_name_or_path == \
        'bert-base-uncased'
    assert isinstance(single_bert_token_classifier_model_specify.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_bert_token_classifier_model_specify.totally_arbitrary == 'arbitrary'

def test_save(single_bert_token_classifier_model_save):
    """Asserts that the expected file exists after call to `BertTokenClassifier.save()``.
    """
    model_filepath = single_bert_token_classifier_model_save

    assert os.path.isfile(model_filepath)

def test_load(single_bert_token_classifier_model, single_bert_token_classifier_model_save,
              dummy_config, dummy_dataset_1):
    """Asserts that the attributes of a BertTokenClassifier object are expected after call to
    `BertTokenClassifier.load()`.
    """
    model_filepath = single_bert_token_classifier_model_save

    single_bert_token_classifier_model.load(model_filepath)

    assert isinstance(single_bert_token_classifier_model,
                      (BertTokenClassifier, BasePyTorchModel, BaseModel))
    # attributes that are passed to __init__
    assert single_bert_token_classifier_model.config is dummy_config
    assert single_bert_token_classifier_model.datasets[0] == dummy_dataset_1
    # other instance attributes
    assert all([isinstance(model, BertForTokenClassification)
                for model in single_bert_token_classifier_model.models])
    assert single_bert_token_classifier_model.embeddings is None
    assert single_bert_token_classifier_model.pretrained_model_name_or_path == 'bert-base-uncased'
    assert isinstance(single_bert_token_classifier_model.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_bert_token_classifier_model.totally_arbitrary == 'arbitrary'
