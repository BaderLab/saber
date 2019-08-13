"""Test suite for the `BertForNER` class (saber.models.bert_for_ner.BertForNER).
"""
import os

from pytorch_transformers import CONFIG_NAME
from pytorch_transformers import WEIGHTS_NAME
from pytorch_transformers import BertForTokenClassification
from pytorch_transformers import BertTokenizer
from pytorch_transformers.optimization import AdamW

from ..constants import PARTITIONS
from ..constants import WORDPIECE
from ..models.base_model import BaseModel
from ..models.bert_for_ner import BertForNER
from ..models.modules.bert_for_token_classification_multi_task import \
    BertForTokenClassificationMultiTask

# TODO: test_prepare_data_for_training_simple and test_predict_simple need to be more rigorous
# TODO (John): Add MT tests when the MT model is implemented
# TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected


class TestBertForNER(object):
    """Collects all unit tests for `saber.models.bert_for_ner.BertForNER`.
    """
    def test_attributes_after_initilization(self,
                                            dummy_config,
                                            conll2003datasetreader_load,
                                            bert_for_ner):
        """Asserts instance attributes are as expected after initialization of a `BertForNER` model.
        """
        assert isinstance(
            bert_for_ner,
            (BertForNER, BaseModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner.config is dummy_config
        assert bert_for_ner.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner.model is None

        assert bert_for_ner.device.type == 'cpu'
        assert bert_for_ner.n_gpus == 0

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner.num_labels == [len(conll2003datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_ner.pretrained_model_name_or_path == 'bert-base-uncased'

        assert bert_for_ner.model_name == 'bert-ner'

    def test_attributes_after_initilization_mt(self,
                                               dummy_config_compound_dataset,
                                               conll2003datasetreader_load,
                                               dummy_dataset_2,
                                               bert_for_ner_mt):
        """Asserts instance attributes are as expected after initialization of a multi-task
        `BertForNER` model.
        """
        assert isinstance(
            bert_for_ner_mt,
            (BertForNER, BaseModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_mt.config is dummy_config_compound_dataset
        assert bert_for_ner_mt.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1
        assert dummy_dataset_2.type_to_idx['ent'][WORDPIECE] == \
            len(dummy_dataset_2.type_to_idx['ent']) - 1
        # Other instance attributes
        assert bert_for_ner_mt.model is None

        assert bert_for_ner_mt.device.type == 'cpu'
        assert bert_for_ner_mt.n_gpus == 0

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_mt.num_labels == [
            len(conll2003datasetreader_load.idx_to_tag['ent']),
            len(dummy_dataset_2.idx_to_tag['ent']),
        ]

        assert bert_for_ner_mt.pretrained_model_name_or_path == 'bert-base-uncased'
        assert bert_for_ner_mt.model_name == 'bert-ner'

    def test_specify(self,
                     dummy_config,
                     conll2003datasetreader_load,
                     bert_for_ner_specify):
        """Asserts attributes are as expected after call to `BertForNER.specify()`.
        """
        assert isinstance(
            bert_for_ner_specify,
            (BertForNER, BaseModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_specify.config is dummy_config
        assert bert_for_ner_specify.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner_specify.device.type == 'cpu'
        assert bert_for_ner_specify.n_gpus == 0
        assert bert_for_ner_specify.pretrained_model_name_or_path == 'bert-base-uncased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_specify.num_labels == \
            [len(conll2003datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_ner_specify.model_name == 'bert-ner'

        # Model and tokenizer
        assert isinstance(
            bert_for_ner_specify.model, (BertForTokenClassificationMultiTask,
                                         BertForTokenClassification)
        )
        assert isinstance(bert_for_ner_specify.tokenizer, BertTokenizer)

    def test_specify_mt(self,
                        dummy_config_compound_dataset,
                        conll2003datasetreader_load,
                        dummy_dataset_2,
                        bert_for_ner_specify_mt):
        """Asserts attributes are as expected after call to `BertForNER.specify()` for a multi-task
        model.
        """
        assert isinstance(
            bert_for_ner_specify_mt,
            (BertForNER, BaseModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_specify_mt.config is dummy_config_compound_dataset
        assert bert_for_ner_specify_mt.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1
        assert dummy_dataset_2.type_to_idx['ent'][WORDPIECE] == \
            len(dummy_dataset_2.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner_specify_mt.device.type == 'cpu'
        assert bert_for_ner_specify_mt.n_gpus == 0

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_specify_mt.num_labels == [
            len(conll2003datasetreader_load.idx_to_tag['ent']),
            len(dummy_dataset_2.idx_to_tag['ent']),
        ]

        assert bert_for_ner_specify_mt.pretrained_model_name_or_path == \
            'bert-base-uncased'
        assert bert_for_ner_specify_mt.model_name == 'bert-ner'

        # Model and tokenizer
        assert isinstance(
            bert_for_ner_specify_mt.model, (BertForTokenClassificationMultiTask,
                                            BertForTokenClassification)
        )
        assert bert_for_ner_specify_mt.model.num_labels == \
            bert_for_ner_specify_mt.num_labels
        assert isinstance(bert_for_ner_specify_mt.tokenizer, BertTokenizer)

    def test_save(self, bert_for_ner_save):
        """Asserts that the expected file(s) exists after call to `BertForNER.save()`.
        """
        _, save_dir = bert_for_ner_save

        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(save_dir, CONFIG_NAME)
        output_vocab_file = os.path.join(save_dir, 'vocab.txt')

        assert os.path.isfile(output_model_file)
        assert os.path.isfile(output_config_file)
        assert os.path.isfile(output_vocab_file)

    def test_save_mt(self, bert_for_ner_save_mt):
        """Asserts that the expected file(s) exists after call to `BertForNER.save()` for a
        multi-task smodel.
        """
        _, save_dir = bert_for_ner_save_mt

        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(save_dir, CONFIG_NAME)
        output_vocab_file = os.path.join(save_dir, 'vocab.txt')

        assert os.path.isfile(output_model_file)
        assert os.path.isfile(output_config_file)
        assert os.path.isfile(output_vocab_file)

    def test_load(self, bert_for_ner, bert_for_ner_save):
        """Asserts that the attributes of a BertForNER object are expected after call to
        `BertForNER.load()`.
        """
        model, save_dir = bert_for_ner_save

        expected_pretrained_model_name_or_path = model.pretrained_model_name_or_path
        expected_num_labels = model.num_labels

        bert_for_ner.load(save_dir)

        actual_pretrained_model_name_or_path = bert_for_ner.pretrained_model_name_or_path
        actual_num_labels = bert_for_ner.num_labels

        assert bert_for_ner.device.type == 'cpu'
        assert bert_for_ner.n_gpus == 0

        assert isinstance(
            bert_for_ner.model, (BertForTokenClassificationMultiTask, BertForTokenClassification)
        )
        assert isinstance(bert_for_ner.tokenizer, BertTokenizer)

        assert expected_pretrained_model_name_or_path == actual_pretrained_model_name_or_path
        assert expected_num_labels == actual_num_labels

    def test_load_mt(self, bert_for_ner_mt, bert_for_ner_save_mt):
        """Asserts that the attributes of a BertForNER object are expected after call to
        `BertForNER.load()` for a multi-task model.
        """
        model, model_filepath = bert_for_ner_save_mt

        expected_pretrained_model_name_or_path = model.pretrained_model_name_or_path
        expected_num_labels = model.num_labels

        bert_for_ner_mt.load(model_filepath)

        actual_pretrained_model_name_or_path = bert_for_ner_mt.pretrained_model_name_or_path
        actual_num_labels = bert_for_ner_mt.num_labels

        assert bert_for_ner_mt.device.type == 'cpu'
        assert bert_for_ner_mt.n_gpus == 0

        assert isinstance(
            bert_for_ner_mt.model, (BertForTokenClassificationMultiTask, BertForTokenClassification)
        )
        assert isinstance(bert_for_ner_mt.tokenizer, BertTokenizer)

        assert expected_pretrained_model_name_or_path == actual_pretrained_model_name_or_path
        assert expected_num_labels == actual_num_labels

    # TODO (John): This is a poor excuse for a test
    def test_prepare_data_for_training(self, bert_for_ner_specify):
        """Asserts that the dictionaries returned by `BertForNER.prepare_data_for_training()`
        contain the expected keys.
        """
        training_data = bert_for_ner_specify.prepare_data_for_training()

        for data in training_data:
            for fold in data:
                assert all('x' in fold[p] and 'y' in fold[p] for p in PARTITIONS)

    # TODO (John): This is a poor excuse for a test
    def test_prepare_data_for_training_mt(self, bert_for_ner_specify_mt):
        """Asserts that the dictionaries returned by `BertForNER.prepare_data_for_training()`
        contain the expected keys.
        """
        training_data = bert_for_ner_specify_mt.prepare_data_for_training()

        for data in training_data:
            for fold in data:
                assert all('x' in fold[p] and 'y' in fold[p] for p in PARTITIONS)

    def test_train(self, bert_for_ner_specify):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_ner_specify.config.epochs = 1
        bert_for_ner_specify.train()

    def test_train_mt(self, bert_for_ner_specify_mt):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_ner_specify_mt.config.epochs = 1
        bert_for_ner_specify_mt.train()

    def test_predict(self, bert_for_ner_specify):
        """Asserts that the shape and labels of the predictions returned by `BertForNER.predict()`
        are as expected.
        """
        tokens = [
            ['This', 'is', 'a', 'test', '.'],
            ['With', 'two', 'sentences', '.']
        ]
        valid_types = bert_for_ner_specify.datasets[0].type_to_idx['ent']

        actual = bert_for_ner_specify.predict(tokens)

        for sent, actual_ in zip(tokens, actual):
            assert len(sent) == len(actual_)
            # Can't test exact label seq because it is stochastic, so check all preds are valid
            assert all(label in valid_types for label in actual_)

    def test_predict_mt(self, bert_for_ner_specify_mt):
        """Asserts that the shape and labels of the predictions returned by `BertForNER.predict()`
        are as expected for a multi-task model.
        """
        tokens = [
            ['This', 'is', 'a', 'test', '.'],
            ['With', 'two', 'sentences', '.']
        ]
        valid_types = bert_for_ner_specify_mt.datasets[0].type_to_idx['ent']

        actual = bert_for_ner_specify_mt.predict(tokens)

        for model_output in actual:
            for sent, actual_ in zip(tokens, actual):
                assert len(sent) == len(actual_)
                # Can't test exact label seq because it is stochastic, so check all preds are valid
                assert all(label in valid_types for label in actual_)

    def test_prepare_optimizers(self, bert_for_ner_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNER.prepare_optimizers()`.
        """
        actual = bert_for_ner_specify.prepare_optimizers()

        assert all(isinstance(opt, AdamW) for opt in actual)

    def test_prepare_optimizers_mt(self, bert_for_ner_specify_mt):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNER.prepare_optimizers()` for a multi-task model.
        """
        actual = bert_for_ner_specify_mt.prepare_optimizers()

        assert all(isinstance(opt, AdamW) for opt in actual)
