"""All unit tests for BertForNER (saber/models/BertForNER.py).
"""
import numpy as np
import os
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification

from .. import constants
from pytorch_pretrained_bert.optimization import BertAdam
from ..models.base_model import BaseModel, BasePyTorchModel
from ..models.bert_for_ner import BertForNER
from ..models.bert_for_token_classification_multi_task import BertForTokenClassificationMultiTask
from .resources.constants import *

# TODO: test_prepare_data_for_training_simple and test_predict_simple need to be more rigorous


class TestBertForNER(object):
    """Collects all unit tests for `saber.models.bert_for_ner.BertForNER`.
    """
    def test_attributes_after_init(self, dummy_config, dummy_dataset_1, bert_for_ner_model):
        """Asserts instance attributes are as expected after initialization of a `BertForNER` model.
        """
        assert isinstance(
            bert_for_ner_model,
            (BertForNER, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_model.config is dummy_config
        assert bert_for_ner_model.datasets == [dummy_dataset_1]

        # Other instance attributes
        assert bert_for_ner_model.model is None
        assert bert_for_ner_model.embeddings is None
        assert bert_for_ner_model.device.type == 'cpu'
        assert bert_for_ner_model.n_gpus == 0
        assert bert_for_ner_model.pretrained_model_name_or_path == 'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_model.num_labels == [len(dummy_dataset_1.idx_to_tag) + 1]
        assert bert_for_ner_model.model_name == 'bert-ner'

        # Test that we can pass arbitrary keyword arguments
        assert bert_for_ner_model.totally_arbitrary == 'arbitrary'

    def test_attributes_after_init_mt(self,
                                      dummy_config,
                                      dummy_dataset_1,
                                      dummy_dataset_2,
                                      mt_bert_for_ner_model):
        """Asserts instance attributes are as expected after initialization of a multi-task
        `BertForNER` model.
        """
        assert isinstance(
            mt_bert_for_ner_model,
            (BertForNER, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert mt_bert_for_ner_model.config is dummy_config
        assert mt_bert_for_ner_model.datasets == [dummy_dataset_1, dummy_dataset_2]
        assert mt_bert_for_ner_model.datasets[1] is dummy_dataset_2
        # Other instance attributes
        assert mt_bert_for_ner_model.model is None
        assert mt_bert_for_ner_model.embeddings is None
        assert mt_bert_for_ner_model.device.type == 'cpu'
        assert mt_bert_for_ner_model.n_gpus == 0
        assert mt_bert_for_ner_model.pretrained_model_name_or_path == 'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert mt_bert_for_ner_model.num_labels == [
            len(dummy_dataset_1.idx_to_tag) + 1,
            len(dummy_dataset_2.idx_to_tag) + 1,
        ]
        assert mt_bert_for_ner_model.model_name == 'bert-ner'

        # Test that we can pass arbitrary keyword arguments
        assert mt_bert_for_ner_model.totally_arbitrary == 'arbitrary'

    def test_attributes_after_specify(self,
                                      dummy_config,
                                      dummy_dataset_1,
                                      bert_for_ner_model_specify):
        """Asserts attributes are as expected after call to `BertForNER.specify()`.
        """
        assert isinstance(
            bert_for_ner_model_specify,
            (BertForNER, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_model_specify.config is dummy_config
        assert bert_for_ner_model_specify.datasets == [dummy_dataset_1]

        # Other instance attributes
        assert bert_for_ner_model_specify.embeddings is None
        assert bert_for_ner_model_specify.device.type == 'cpu'
        assert bert_for_ner_model_specify.n_gpus == 0
        assert bert_for_ner_model_specify.pretrained_model_name_or_path == 'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_model_specify.num_labels == [len(dummy_dataset_1.idx_to_tag) + 1]
        assert bert_for_ner_model_specify.model_name == 'bert-ner'

        # Model and tokenizer
        assert isinstance(
            bert_for_ner_model_specify.model, (BertForTokenClassificationMultiTask,
                                               BertForTokenClassification)
        )
        assert bert_for_ner_model_specify.model.num_labels == \
            bert_for_ner_model_specify.num_labels
        assert isinstance(bert_for_ner_model_specify.tokenizer, BertTokenizer)

        # Test that we can pass arbitrary keyword arguments
        assert bert_for_ner_model_specify.totally_arbitrary == 'arbitrary'

    def test_attributes_after_specify_mt(self,
                                         dummy_config,
                                         dummy_dataset_1,
                                         dummy_dataset_2,
                                         mt_bert_for_ner_model_specify):
        """Asserts attributes are as expected after call to `BertForNER.specify()` for a multi-task
        model.
        """
        assert isinstance(
            mt_bert_for_ner_model_specify,
            (BertForNER, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert mt_bert_for_ner_model_specify.config is dummy_config
        assert mt_bert_for_ner_model_specify.datasets == [dummy_dataset_1, dummy_dataset_2]

        # Other instance attributes
        assert mt_bert_for_ner_model_specify.embeddings is None
        assert mt_bert_for_ner_model_specify.device.type == 'cpu'
        assert mt_bert_for_ner_model_specify.n_gpus == 0
        assert mt_bert_for_ner_model_specify.pretrained_model_name_or_path == \
            'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert mt_bert_for_ner_model_specify.num_labels == [
            len(dummy_dataset_1.idx_to_tag) + 1,
            len(dummy_dataset_2.idx_to_tag) + 1,
        ]
        assert mt_bert_for_ner_model_specify.model_name == 'bert-ner'

        # Model and tokenizer
        assert isinstance(
            mt_bert_for_ner_model_specify.model, (BertForTokenClassificationMultiTask,
                                                  BertForTokenClassification)
        )
        assert mt_bert_for_ner_model_specify.model.num_labels == \
            mt_bert_for_ner_model_specify.num_labels
        assert isinstance(mt_bert_for_ner_model_specify.tokenizer, BertTokenizer)

        # Test that we can pass arbitrary keyword arguments
        assert mt_bert_for_ner_model_specify.totally_arbitrary == 'arbitrary'

    def test_save(self, bert_for_ner_model_save):
        """Asserts that the expected file exists after call to `BertForNER.save()`.
        """
        _, model_filepath = bert_for_ner_model_save

        assert os.path.isfile(model_filepath)

    def test_save_mt(self, bert_for_ner_model_save):
        """Asserts that the expected file exists after call to `BertForNER.save()` for a multi-task
        model.
        """
        _, model_filepath = bert_for_ner_model_save

        assert os.path.isfile(model_filepath)

    def test_load(self, bert_for_ner_model, bert_for_ner_model_save):
        """Asserts that the attributes of a BertForNER object are expected after call to
        `BertForNER.load()`.
        """
        model, model_filepath = bert_for_ner_model_save
        expected_num_labels = model.num_labels

        bert_for_ner_model.load(model_filepath)

        assert isinstance(
            bert_for_ner_model.model, (BertForTokenClassificationMultiTask,
                                       BertForTokenClassification)
        )
        assert bert_for_ner_model.model.num_labels == expected_num_labels
        assert isinstance(bert_for_ner_model.tokenizer, BertTokenizer)

    def test_load_mt(self, mt_bert_for_ner_model, mt_bert_for_ner_model_save):
        """Asserts that the attributes of a BertForNER object are expected after call to
        `BertForNER.load()` for a multi-task model.
        """
        model, model_filepath = mt_bert_for_ner_model_save
        expected_num_labels = model.num_labels

        mt_bert_for_ner_model.load(model_filepath)

        assert isinstance(
            mt_bert_for_ner_model.model, (BertForTokenClassificationMultiTask,
                                          BertForTokenClassification)
        )
        assert mt_bert_for_ner_model.model.num_labels == expected_num_labels
        assert isinstance(mt_bert_for_ner_model.tokenizer, BertTokenizer)

    def test_prepare_data_for_training(self, bert_for_ner_model_specify):
        """Asserts that the dictionaries returned by `BertForNER.prepare_data_for_training()`
        contain the expected keys.
        """
        training_data = bert_for_ner_model_specify.prepare_data_for_training()

        assert all('x_{}'.format(part) in data and 'y_{}'.format(part) in data
                   for data in training_data for part in constants.PARTITIONS)

    def test_train(self, bert_for_ner_model_specify):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_ner_model_specify.config.epochs = 1
        bert_for_ner_model_specify.train()

        # This won't print anything unless the test fails
        print('The training loop is likely broken.')

        assert True

    '''Need to fix the StatisticsError that is thrown.
    def test_cross_validation(bert_for_ner_model_specify):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_ner_model_specify.config.epochs = 1
        bert_for_ner_model_specify.cross_validation()

        # This won't print anything unless the test fails
        print('The cross validation training loop is likely broken')

        assert True
    '''

    def test_predict(self, bert_for_ner_model_specify):
        """Asserts that the shape of the predictions returned by `BertForNER.predict()` are as
        expected.
        """
        input_ = np.asarray([['This', 'is', 'a', 'test', '.']])

        X_actual, y_pred_actual = bert_for_ner_model_specify.predict(input_)

        assert isinstance(X_actual, np.ndarray)
        assert isinstance(y_pred_actual, np.ndarray)
        # The expected shape of the processed input and predictions is (1, constants.MAX_SENT_LEN)
        # as there is 1 sentence in input_
        assert X_actual.shape == y_pred_actual.argmax(axis=-1).shape == (1, constants.MAX_SENT_LEN)

    def test_prepare_optimizers(self, bert_for_ner_model_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNER.prepare_optimizers()`.
        """
        actual = bert_for_ner_model_specify.prepare_optimizers()

        assert all(isinstance(opt, BertAdam) for opt in actual)

    def test_prepare_optimizers_mt(self, mt_bert_for_ner_model_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNER.prepare_optimizers()` for a multi-task model.
        """
        actual = mt_bert_for_ner_model_specify.prepare_optimizers()

        assert all(isinstance(opt, BertAdam) for opt in actual)
