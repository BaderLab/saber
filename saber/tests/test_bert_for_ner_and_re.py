"""Test suite for the `BertForNERAndRE` class
(saber.models.bert_for_ner_and_re.BertForNERAndRE).
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
from ..models.bert_for_ner_and_re import BertForNERAndRE
from ..models.modules.bert_for_entity_and_relation_extraction import \
    BertForEntityAndRelationExtraction

# TODO: test_prepare_data_for_training_simple and test_predict_simple need to be more rigorous
# TODO (John): Add MT tests when the MT model is implemented
# TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected


class TestBertForNERAndRE(object):
    """Collects all unit tests for `saber.models.bert_for_ner_and_re.BertForNERAndRE`.
    """
    def test_attributes_after_initilization(self,
                                            dummy_config,
                                            conll2004datasetreader_load,
                                            bert_for_ner_and_re):
        """Asserts instance attributes are as expected after initialization of a
        `BertForNERAndRE` model.
        """
        assert isinstance(
            bert_for_ner_and_re,
            (BertForNERAndRE, BaseModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_and_re.config is dummy_config
        assert bert_for_ner_and_re.datasets[0] is conll2004datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2004datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2004datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner_and_re.model is None

        assert bert_for_ner_and_re.device.type == 'cpu'
        assert bert_for_ner_and_re.n_gpus == 0

        assert bert_for_ner_and_re.num_ent_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_ner_and_re.num_rel_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['rel'])]

        assert bert_for_ner_and_re.pretrained_model_name_or_path == 'bert-base-uncased'
        assert bert_for_ner_and_re.model_name == 'bert-ner-re'

    def test_attributes_after_initilization_mt(self,
                                               dummy_config,
                                               conll2004datasetreader_load,
                                               bert_for_ner_and_re):
        """Asserts instance attributes are as expected after initialization of a multi-task
        `BertForNERAndRE` model.
        """
        pass

    def test_specify(self,
                     dummy_config,
                     conll2004datasetreader_load,
                     bert_for_ner_and_re_specify):
        """Asserts attributes are as expected after call to `BertForNERAndRE.specify()`.
        """
        assert isinstance(
            bert_for_ner_and_re_specify,
            (BertForNERAndRE, BaseModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_and_re_specify.config is dummy_config
        assert bert_for_ner_and_re_specify.datasets[0] is conll2004datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2004datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2004datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner_and_re_specify.device.type == 'cpu'
        assert bert_for_ner_and_re_specify.n_gpus == 0
        assert bert_for_ner_and_re_specify.pretrained_model_name_or_path == \
            'bert-base-uncased'

        assert bert_for_ner_and_re_specify.num_ent_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_ner_and_re_specify.num_rel_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['rel'])]
        assert bert_for_ner_and_re_specify.model_name == 'bert-ner-re'

        # Model and tokenizer
        assert isinstance(
            bert_for_ner_and_re_specify.model,
            (BertForEntityAndRelationExtraction, BertForTokenClassification)
        )
        assert isinstance(bert_for_ner_and_re_specify.tokenizer, BertTokenizer)

    def test_specify_mt(self,
                        dummy_config,
                        conll2004datasetreader_load,
                        bert_for_ner_and_re_specify):
        """Asserts attributes are as expected after call to `BertForNERAndRE.specify()` for a
        multi-task model.
        """
        pass

    def test_save(self, bert_for_ner_and_re_save):
        """Asserts that the expected file(s) exists after call to `BertForNERAndRE.save()`.
        """
        _, save_dir = bert_for_ner_and_re_save

        output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(save_dir, CONFIG_NAME)
        output_vocab_file = os.path.join(save_dir, 'vocab.txt')

        assert os.path.isfile(output_model_file)
        assert os.path.isfile(output_config_file)
        assert os.path.isfile(output_vocab_file)

    def test_save_mt(self, bert_for_ner_and_re_save):
        """Asserts that the expected file(s) exists after call to `BertForNERAndRE.save()` for
        a multi-task model.
        """
        pass

    def test_load(self, bert_for_ner_and_re, bert_for_ner_and_re_save):
        """Asserts that the attributes of a `BertForNERAndRE` object are expected after call to
        `BertForNERAndRE.load()`.
        """
        model, save_dir = bert_for_ner_and_re_save

        expected_pretrained_model_name_or_path = model.pretrained_model_name_or_path
        expected_num_ent_labels = model.num_ent_labels
        expected_num_rel_labels = model.num_rel_labels

        bert_for_ner_and_re.load(save_dir)

        actual_pretrained_model_name_or_path = \
            bert_for_ner_and_re.pretrained_model_name_or_path
        actual_num_ent_labels = bert_for_ner_and_re.num_ent_labels
        actual_num_rel_labels = bert_for_ner_and_re.num_rel_labels

        assert bert_for_ner_and_re.device.type == 'cpu'
        assert bert_for_ner_and_re.n_gpus == 0

        assert isinstance(
            bert_for_ner_and_re.model,
            (BertForEntityAndRelationExtraction, BertForTokenClassification)
        )
        assert isinstance(bert_for_ner_and_re.tokenizer, BertTokenizer)

        assert expected_pretrained_model_name_or_path == actual_pretrained_model_name_or_path
        assert expected_num_ent_labels == actual_num_ent_labels
        assert expected_num_rel_labels == actual_num_rel_labels

    def test_load_mt(self):
        """Asserts that the attributes of a BertForNERAndRE object are expected after call to
        `BertForNERAndRE.load()` for a multi-task model.
        """
        pass

    # TODO (John): This is a poor excuse for a test
    def test_prepare_data_for_training(self, bert_for_ner_specify):
        """Asserts that the dictionaries returned by `BertForNER.prepare_data_for_training()`
        contain the expected keys.
        """
        training_data = bert_for_ner_specify.prepare_data_for_training()

        for data in training_data:
            for fold in data:
                assert all('x' in fold[p] and 'y' in fold[p] for p in PARTITIONS)

    def test_train(self, bert_for_ner_and_re_specify):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_ner_and_re_specify.config.epochs = 1
        bert_for_ner_and_re_specify.train()

    # TODO (John): This doesn't account for relation predictions
    def test_predict(self, bert_for_ner_and_re_specify):
        """Asserts that the shape and labels of the predictions returned by `BertForNERAndRE.predict()`
        are as expected.
        """
        tokens = [
            ['This', 'is', 'a', 'test', '.'],
            ['With', 'two', 'sentences', '.']
        ]
        valid_ner_labels = bert_for_ner_and_re_specify.datasets[0].type_to_idx['ent']

        actual_ner_preds, _, = bert_for_ner_and_re_specify.predict(tokens)

        for sent, actual_ner in zip(tokens, actual_ner_preds):
            assert len(sent) == len(actual_ner)
            # Can't test exact label seq because it is stochastic, so check all preds are valid
            assert all(label in valid_ner_labels for label in actual_ner)

    def test_predict_mt(self):
        """Asserts that the shape and labels of the predictions returned by
        `BertForNERAndRE.predict()` are as expected for a multi-task model.
        """
        pass

    def test_prepare_optimizers(self, bert_for_ner_and_re_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNERAndRE._prepare_optimizers()`.
        """
        actual = bert_for_ner_and_re_specify._prepare_optimizers()

        assert all(isinstance(opt, AdamW) for opt in actual)

    def test_prepare_optimizers_mt(self):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNERAndRE._prepare_optimizers()` for a multi-task model.
        """
        pass
