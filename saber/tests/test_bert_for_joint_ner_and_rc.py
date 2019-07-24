import os

from pytorch_pretrained_bert import BertForTokenClassification
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

from ..constants import PARTITIONS
from ..constants import WORDPIECE
from ..models.base_model import BaseModel
from ..models.base_model import BasePyTorchModel
from ..models.bert_for_joint_ner_and_rc import BertForJointNERAndRE
from ..models.modules.bert_for_joint_entity_and_relation_classification import \
    BertForJointEntityAndRelationExtraction

# TODO: test_prepare_data_for_training_simple and test_predict_simple need to be more rigorous
# TODO (John): Add MT tests when the MT model is implemented
# TODO (johngiorgi): Test that saving loading from CPU/GPU works as expected


class TestBertForJointNERAndRE(object):
    """Collects all unit tests for `saber.models.bert_for_joint_ner_and_rc.BertForJointNERAndRE`.
    """
    def test_attributes_after_init(self,
                                   dummy_config,
                                   conll2004datasetreader_load,
                                   bert_for_joint_ner_and_rc_model):
        """Asserts instance attributes are as expected after initialization of a
        `BertForJointNERAndRE` model.
        """
        assert isinstance(
            bert_for_joint_ner_and_rc_model,
            (BertForJointNERAndRE, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_joint_ner_and_rc_model.config is dummy_config
        assert bert_for_joint_ner_and_rc_model.datasets[0] is conll2004datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2004datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2004datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_joint_ner_and_rc_model.model is None
        assert bert_for_joint_ner_and_rc_model.embeddings is None
        assert bert_for_joint_ner_and_rc_model.device.type == 'cpu'
        assert bert_for_joint_ner_and_rc_model.n_gpus == 0
        assert bert_for_joint_ner_and_rc_model.pretrained_model_name_or_path == 'bert-base-cased'

        assert bert_for_joint_ner_and_rc_model.num_ent_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_joint_ner_and_rc_model.num_rel_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['rel'])]
        assert bert_for_joint_ner_and_rc_model.model_name == 'bert-ner-rc'

        # Test that we can pass arbitrary keyword arguments
        assert bert_for_joint_ner_and_rc_model.totally_arbitrary == 'arbitrary'

    def test_attributes_after_specify(self,
                                      dummy_config,
                                      conll2004datasetreader_load,
                                      bert_for_joint_ner_and_rc_specify):
        """Asserts attributes are as expected after call to `BertForJointNERAndRE.specify()`.
        """
        assert isinstance(
            bert_for_joint_ner_and_rc_specify,
            (BertForJointNERAndRE, BasePyTorchModel, BaseModel)
        )

        # Attributes that are passed to __init__
        assert bert_for_joint_ner_and_rc_specify.config is dummy_config
        assert bert_for_joint_ner_and_rc_specify.datasets[0] is conll2004datasetreader_load
        # Check that intialization has added the wordpiece tag ('X') with correct index
        assert conll2004datasetreader_load.type_to_idx['ent'][WORDPIECE] == \
            len(conll2004datasetreader_load.type_to_idx['ent']) - 1

        # Other instance attributes
        assert bert_for_joint_ner_and_rc_specify.embeddings is None
        assert bert_for_joint_ner_and_rc_specify.device.type == 'cpu'
        assert bert_for_joint_ner_and_rc_specify.n_gpus == 0
        assert bert_for_joint_ner_and_rc_specify.pretrained_model_name_or_path == \
            'bert-base-cased'

        assert bert_for_joint_ner_and_rc_specify.num_ent_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['ent'])]
        assert bert_for_joint_ner_and_rc_specify.num_rel_labels == \
            [len(conll2004datasetreader_load.idx_to_tag['rel'])]
        assert bert_for_joint_ner_and_rc_specify.model_name == 'bert-ner-rc'

        # Model and tokenizer
        assert isinstance(
            bert_for_joint_ner_and_rc_specify.model,
            (BertForJointEntityAndRelationExtraction, BertForTokenClassification)
        )
        assert isinstance(bert_for_joint_ner_and_rc_specify.tokenizer, BertTokenizer)

        # Test that we can pass arbitrary keyword arguments
        assert bert_for_joint_ner_and_rc_specify.totally_arbitrary == 'arbitrary'

    def test_save(self, bert_for_joint_ner_and_rc_save):
        """Asserts that the expected file(s) exists after call to `BertForJointNERAndRE.save()`.
        """
        _, model_filepath = bert_for_joint_ner_and_rc_save

        assert os.path.isfile(model_filepath)

    def test_load(self, bert_for_joint_ner_and_rc_model, bert_for_joint_ner_and_rc_save):
        """Asserts that the attributes of a `BertForJointNERAndRE` object are expected after call to
        `BertForJointNERAndRE.load()`.
        """
        model, model_filepath = bert_for_joint_ner_and_rc_save

        expected_pretrained_model_name_or_path = model.pretrained_model_name_or_path
        expected_num_ent_labels = model.num_ent_labels
        expected_num_rel_labels = model.num_rel_labels

        bert_for_joint_ner_and_rc_model.load(model_filepath)

        actual_pretrained_model_name_or_path = \
            bert_for_joint_ner_and_rc_model.pretrained_model_name_or_path
        actual_num_ent_labels = bert_for_joint_ner_and_rc_model.num_ent_labels
        actual_num_rel_labels = bert_for_joint_ner_and_rc_model.num_rel_labels

        assert bert_for_joint_ner_and_rc_model.device.type == 'cpu'
        assert bert_for_joint_ner_and_rc_model.n_gpus == 0

        assert isinstance(
            bert_for_joint_ner_and_rc_model.model,
            (BertForJointEntityAndRelationExtraction, BertForTokenClassification)
        )
        assert isinstance(bert_for_joint_ner_and_rc_model.tokenizer, BertTokenizer)

        assert expected_pretrained_model_name_or_path == actual_pretrained_model_name_or_path
        assert expected_num_ent_labels == actual_num_ent_labels
        assert expected_num_rel_labels == actual_num_rel_labels

    def test_load_mt(self):
        """Asserts that the attributes of a BertForJointNERAndRE object are expected after call to
        `BertForJointNERAndRE.load()` for a multi-task model.
        """
        pass

    # TODO (John): This is a poor excuse for a test
    def test_prepare_data_for_training(self, bert_for_ner_model_specify):
        """Asserts that the dictionaries returned by `BertForNER.prepare_data_for_training()`
        contain the expected keys.
        """
        training_data = bert_for_ner_model_specify.prepare_data_for_training()

        for data in training_data:
            for fold in data:
                assert all('x' in fold[p] and 'y' in fold[p] for p in PARTITIONS)

    def test_train(self, bert_for_joint_ner_and_rc_specify):
        """This test does not actually assert anything (which is surely bad practice) but at the
        very least, it will fail if training was unsuccesful and therefore alert us when a code
        change has broke the training loop.
        """
        bert_for_joint_ner_and_rc_specify.config.epochs = 1
        bert_for_joint_ner_and_rc_specify.train()

    # TODO (John): This doesn't account for relation predictions
    def test_predict(self, bert_for_joint_ner_and_rc_specify):
        """Asserts that the shape and labels of the predictions returned by `BertForJointNERAndRE.predict()`
        are as expected.
        """
        tokens = [
            ['This', 'is', 'a', 'test', '.'],
            ['With', 'two', 'sentences', '.']
        ]
        valid_ner_labels = bert_for_joint_ner_and_rc_specify.datasets[0].type_to_idx['ent']

        actual_ner_preds, _, = bert_for_joint_ner_and_rc_specify.predict(tokens)

        for sent, actual_ner in zip(tokens, actual_ner_preds):
            assert len(sent) == len(actual_ner)
            # Can't test exact label seq because it is stochastic, so check all preds are valid
            assert all(label in valid_ner_labels for label in actual_ner)

    def test_predict_mt(self):
        """Asserts that the shape and labels of the predictions returned by
        `BertForJointNERAndRE.predict()` are as expected for a multi-task model.
        """
        pass

    def test_prepare_optimizers(self, bert_for_joint_ner_and_rc_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForJointNERAndRE.prepare_optimizers()`.
        """
        actual = bert_for_joint_ner_and_rc_specify.prepare_optimizers()

        assert all(isinstance(opt, BertAdam) for opt in actual)

    def test_prepare_optimizers_mt(self):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForJointNERAndRE.prepare_optimizers()` for a multi-task model.
        """
        pass
