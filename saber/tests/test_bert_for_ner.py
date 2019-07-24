import os

from pytorch_pretrained_bert import BertForTokenClassification
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from ..constants import PARTITIONS
from ..constants import WORDPIECE
from ..models.base_model import BaseModel
from ..models.base_model import BasePyTorchModel
from ..models.bert_for_ner import BertForNER
from ..models.modules.bert_for_token_classification_multi_task import \
    BertForTokenClassificationMultiTask

# TODO: test_prepare_data_for_training_simple and test_predict_simple need to be more rigorous
# TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected


class TestBertForNER(object):
    """Collects all unit tests for `saber.models.bert_for_ner.BertForNER`.
    """
    def test_attributes_after_init(self,
                                   dummy_config,
                                   conll2003datasetreader_load,
                                   bert_for_ner_model):
        """Asserts instance attributes are as expected after initialization of a `BertForNER` model.
        """
        assert isinstance(
            bert_for_ner_model,
            (BertForNER, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_model.config is dummy_config
        assert bert_for_ner_model.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner_model.model is None
        assert bert_for_ner_model.embeddings is None
        assert bert_for_ner_model.device.type == 'cpu'
        assert bert_for_ner_model.n_gpus == 0
        assert bert_for_ner_model.pretrained_model_name_or_path == 'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_model.num_labels == [len(conll2003datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_ner_model.model_name == 'bert-ner'

        # Test that we can pass arbitrary keyword arguments
        assert bert_for_ner_model.totally_arbitrary == 'arbitrary'

    def test_attributes_after_init_mt(self,
                                      dummy_config_compound_dataset,
                                      conll2003datasetreader_load,
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
        assert mt_bert_for_ner_model.config is dummy_config_compound_dataset
        assert mt_bert_for_ner_model.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1
        assert dummy_dataset_2.type_to_idx['ent'][WORDPIECE] == \
            len(dummy_dataset_2.type_to_idx['ent']) - 1
        # Other instance attributes
        assert mt_bert_for_ner_model.model is None
        assert mt_bert_for_ner_model.embeddings is None
        assert mt_bert_for_ner_model.device.type == 'cpu'
        assert mt_bert_for_ner_model.n_gpus == 0
        assert mt_bert_for_ner_model.pretrained_model_name_or_path == 'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert mt_bert_for_ner_model.num_labels == [
            len(conll2003datasetreader_load.idx_to_tag['ent']),
            len(dummy_dataset_2.idx_to_tag['ent']),
        ]
        assert mt_bert_for_ner_model.model_name == 'bert-ner'

        # Test that we can pass arbitrary keyword arguments
        assert mt_bert_for_ner_model.totally_arbitrary == 'arbitrary'

    def test_attributes_after_specify(self,
                                      dummy_config,
                                      conll2003datasetreader_load,
                                      bert_for_ner_model_specify):
        """Asserts attributes are as expected after call to `BertForNER.specify()`.
        """
        assert isinstance(
            bert_for_ner_model_specify,
            (BertForNER, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_ner_model_specify.config is dummy_config
        assert bert_for_ner_model_specify.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_ner_model_specify.embeddings is None
        assert bert_for_ner_model_specify.device.type == 'cpu'
        assert bert_for_ner_model_specify.n_gpus == 0
        assert bert_for_ner_model_specify.pretrained_model_name_or_path == 'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert bert_for_ner_model_specify.num_labels == \
            [len(conll2003datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_ner_model_specify.model_name == 'bert-ner'

        # Model and tokenizer
        assert isinstance(
            bert_for_ner_model_specify.model, (BertForTokenClassificationMultiTask,
                                               BertForTokenClassification)
        )
        assert isinstance(bert_for_ner_model_specify.tokenizer, BertTokenizer)

        # Test that we can pass arbitrary keyword arguments
        assert bert_for_ner_model_specify.totally_arbitrary == 'arbitrary'

    def test_attributes_after_specify_mt(self,
                                         dummy_config_compound_dataset,
                                         conll2003datasetreader_load,
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
        assert mt_bert_for_ner_model_specify.config is dummy_config_compound_dataset
        assert mt_bert_for_ner_model_specify.datasets[0] is conll2003datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2003datasetreader_load.type_to_idx['ent']) - 1
        assert dummy_dataset_2.type_to_idx['ent'][WORDPIECE] == \
            len(dummy_dataset_2.type_to_idx['ent']) - 1

        # Other instance attributes
        assert mt_bert_for_ner_model_specify.embeddings is None
        assert mt_bert_for_ner_model_specify.device.type == 'cpu'
        assert mt_bert_for_ner_model_specify.n_gpus == 0
        assert mt_bert_for_ner_model_specify.pretrained_model_name_or_path == \
            'bert-base-cased'

        # +1 necessary to account for 'X' tag introduced by wordpiece tokenization
        assert mt_bert_for_ner_model_specify.num_labels == [
            len(conll2003datasetreader_load.idx_to_tag['ent']),
            len(dummy_dataset_2.idx_to_tag['ent']),
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
        """Asserts that the expected file(s) exists after call to `BertForNER.save()`.
        """
        _, model_filepath = bert_for_ner_model_save

        assert os.path.isfile(model_filepath)

    def test_save_mt(self, bert_for_ner_model_save):
        """Asserts that the expected file(s) exists after call to `BertForNER.save()` for a
        multi-task smodel.
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

    # TODO (John): This is a poor excuse for a test
    def test_prepare_data_for_training(self, bert_for_ner_model_specify):
        """Asserts that the dictionaries returned by `BertForNER.prepare_data_for_training()`
        contain the expected keys.
        """
        training_data = bert_for_ner_model_specify.prepare_data_for_training()

        for data in training_data:
            for fold in data:
                assert all('x' in fold[p] and 'y' in fold[p] for p in PARTITIONS)

    def test_train(self, bert_for_ner_model_specify):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_ner_model_specify.config.epochs = 1
        bert_for_ner_model_specify.train()

    def test_predict(self, bert_for_ner_model_specify):
        """Asserts that the shape and labels of the predictions returned by `BertForNER.predict()`
        are as expected.
        """
        tokens = [
            ['This', 'is', 'a', 'test', '.'],
            ['With', 'two', 'sentences', '.']
        ]
        valid_types = bert_for_ner_model_specify.datasets[0].type_to_idx['ent']

        actual = bert_for_ner_model_specify.predict(tokens)

        for sent, actual_ in zip(tokens, actual):
            assert len(sent) == len(actual_)
            # Can't test exact label seq because it is stochastic, so check all preds are valid
            assert all(label in valid_types for label in actual_)

    def test_predict_mt(self, mt_bert_for_ner_model_specify):
        """Asserts that the shape and labels of the predictions returned by `BertForNER.predict()`
        are as expected for a multi-task model.
        """
        tokens = [
            ['This', 'is', 'a', 'test', '.'],
            ['With', 'two', 'sentences', '.']
        ]
        valid_types = mt_bert_for_ner_model_specify.datasets[0].type_to_idx['ent']

        actual = mt_bert_for_ner_model_specify.predict(tokens)

        for model_output in actual:
            for sent, actual_ in zip(tokens, actual):
                assert len(sent) == len(actual_)
                # Can't test exact label seq because it is stochastic, so check all preds are valid
                assert all(label in valid_types for label in actual_)

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
