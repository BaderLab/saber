"""Any and all unit tests for the BertForTokenClassification 
(saber/models/BertForTokenClassification.py).
"""
import numpy as np
import os
from pytorch_pretrained_bert import BertForTokenClassification, BertTokenizer

from .. import constants
from ..models.base_model import BaseModel, BasePyTorchModel
from ..models.bert_token_classifier import BertTokenClassifier
from .resources.constants import *

# TODO: test_prepare_data_for_training_simple and test_predict_simple need to be more rigorous


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
    assert single_bert_token_classifier_model.model is None
    assert single_bert_token_classifier_model.embeddings is None
    assert single_bert_token_classifier_model.device.type == 'cpu'
    assert single_bert_token_classifier_model.n_gpus == 0
    assert single_bert_token_classifier_model.pretrained_model_name_or_path == 'bert-base-cased'
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
    assert isinstance(single_bert_token_classifier_model_specify.model, BertForTokenClassification)
    assert single_bert_token_classifier_model_specify.embeddings is None
    assert single_bert_token_classifier_model_specify.device.type == 'cpu'
    assert single_bert_token_classifier_model_specify.n_gpus == 0
    assert single_bert_token_classifier_model_specify.pretrained_model_name_or_path == 'bert-base-cased'
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
    assert isinstance(single_bert_token_classifier_model.model, BertForTokenClassification)
    assert single_bert_token_classifier_model.embeddings is None
    assert single_bert_token_classifier_model.device.type == 'cpu'
    assert single_bert_token_classifier_model.n_gpus == 0
    assert single_bert_token_classifier_model.pretrained_model_name_or_path == 'bert-base-cased'
    assert isinstance(single_bert_token_classifier_model.tokenizer, BertTokenizer)
    # test that we can pass arbitrary keyword arguments
    assert single_bert_token_classifier_model.totally_arbitrary == 'arbitrary'


def test_prepare_data_for_training_simple(single_bert_token_classifier_model_specify):
    """Asserts that the dictionaries returned by
    `single_bert_token_classifier_model_specify.prepare_data_for_training()` contain the expected
    keys.
    """
    training_data = single_bert_token_classifier_model_specify.prepare_data_for_training()

    assert all('x_{}'.format(part) in data and 'y_{}'.format(part) in data
               for data in training_data for part in constants.PARTITIONS)


def test_train_simple(single_bert_token_classifier_model_specify):
    """This test does not actually assert anything (which is surely bad practice) but at the very
    least, it will fail if training was unsuccesful and therefore alert us when a code change has
    broke the training loop.
    """
    single_bert_token_classifier_model_specify.config.epochs = 1
    single_bert_token_classifier_model_specify.train()

    # This won't print anything unless the test fails
    print('The training loop is likely broken.')

    assert True


'''Need to fix the StatisticsError that is thrown.
def test_cross_validation(single_bert_token_classifier_model_specify):
    """This test does not actually assert anything (which is surely bad practice) but at the very
    least, it will fail if training was unsuccesful and therefore alert us when a code change has
    broke the training loop.
    """
    single_bert_token_classifier_model_specify.config.epochs = 1
    single_bert_token_classifier_model_specify.cross_validation()

    # This won't print anything unless the test fails
    print('The cross validation training loop is likely broken')

    assert True
'''


def test_predict_simple(single_bert_token_classifier_model_specify):
    """Asserts that the shape of the predictions returned by
    `single_bert_token_classifier_model_specify.predict()` are as expected.
    """
    input_ = np.asarray([['This', 'is', 'a', 'test', '.']])

    X_actual, y_pred_actual = single_bert_token_classifier_model_specify.predict(input_)

    assert isinstance(X_actual, np.ndarray)
    assert isinstance(y_pred_actual, np.ndarray)
    # The expected shape of the processed input and predictions is (1, constants.MAX_SENT_LEN) as
    # there is 1 sentence in input_
    assert X_actual.shape == y_pred_actual.argmax(axis=-1).shape == (1, constants.MAX_SENT_LEN)
