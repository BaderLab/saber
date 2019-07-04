import torch
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

from .. import constants
from ..constants import CLS
from ..constants import PAD
from ..constants import SEP
from ..constants import TOK_MAP_PAD
from ..constants import WORDPIECE
from ..utils import bert_utils


class TestBertUtils(object):
    """Collects all unit tests for `saber.utils.bert_utils`.
    """
    def test_process_dataset_for_bert_conll2003(self, conll2003datasetreader_load, bert_tokenizer):
        """Asserts that `bert_utils.process_dataset_for_bert()` returns the expected values for a
        given `Dataset` object.
        """
        # Needed to check that the WORDPIECE token was added and assigned to the correct index
        num_ent_types = len(conll2003datasetreader_load.type_to_idx['ent'])

        actual = bert_utils.process_dataset_for_bert(conll2003datasetreader_load, bert_tokenizer)

        # Check WORDPIECE token was added to the dataset
        assert conll2003datasetreader_load.type_to_idx['ent'][WORDPIECE] == num_ent_types
        assert conll2003datasetreader_load.idx_to_tag['ent'][num_ent_types] == WORDPIECE

        # Check shape instead of content, as content is checked in different unit tests
        for partition in conll2003datasetreader_load.dataset_folder:
            assert actual[f'x_{partition}'][0].size() == (2, constants.MAX_SENT_LEN)
            assert actual[f'x_{partition}'][1].size() == (2, constants.MAX_SENT_LEN)
            assert actual[f'y_{partition}'].size() == (2, constants.MAX_SENT_LEN)
            assert actual[f'orig_to_tok_map_{partition}'].size() == (2, constants.MAX_SENT_LEN)
            # This is a CoNLL2003 formatted dataset so there should be no rel labels
            assert f'rel_labels_{partition}' not in actual

    def test_process_dataset_for_bert_conll2004(self, conll2004datasetreader_load, bert_tokenizer):
        """Asserts that `bert_utils.process_dataset_for_bert()` returns the expected values for a
        given `Dataset` object.
        """
        # Needed to check that the WORDPIECE token was added and assigned to the correct index
        num_ent_types = len(conll2004datasetreader_load.type_to_idx['ent'])

        actual = bert_utils.process_dataset_for_bert(conll2004datasetreader_load, bert_tokenizer)

        # Check WORDPIECE token was added to the dataset
        assert conll2004datasetreader_load.type_to_idx['ent'][WORDPIECE] == num_ent_types
        assert conll2004datasetreader_load.idx_to_tag['ent'][num_ent_types] == WORDPIECE

        # Check shape instead of content, as content is checked in different unit tests
        for partition in conll2004datasetreader_load.dataset_folder:
            assert actual[f'x_{partition}'][0].size() == (3, constants.MAX_SENT_LEN)
            assert actual[f'x_{partition}'][1].size() == (3, constants.MAX_SENT_LEN)
            assert actual[f'y_{partition}'].size() == (3, constants.MAX_SENT_LEN)
            assert actual[f'orig_to_tok_map_{partition}'].size() == (3, constants.MAX_SENT_LEN)
            assert actual[f'rel_labels_{partition}'] == \
                conll2004datasetreader_load.idx_seq[partition]['rel']

    def test_wordpiece_tokenize_sents(self, bert_tokenizer):
        """Asserts that `bert_utils.wordpiece_tokenize_sents()` returns the expected values for a
        simple input when input argument `labels` is None.
        """
        tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]

        expected_bert_tokens = [
            [CLS, "john", "johan", "##son", "'", "##s",  "house", SEP],
            [CLS, "who", "was", "jim", "henson", "?", SEP]
        ]
        expected_orig_to_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]

        actual_bert_tokens, actual_orig_to_tok_map = \
            bert_utils.wordpiece_tokenize_sents(tokens=tokens,
                                                tokenizer=bert_tokenizer,)

        assert expected_bert_tokens == actual_bert_tokens
        assert expected_orig_to_tok_map == actual_orig_to_tok_map

    def test_wordpiece_tokenize_sents_labels(self, bert_tokenizer):
        """Asserts that `bert_utils.wordpiece_tokenize_sents()` returns the expected values for a
        simple input when input argument `labels` is not None.
        """
        tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]
        labels = [
            ["B-PER", "I-PER", "I-PER",  "O"],
            ["O", "O", "B-PER", "I-PER",  "O"]
        ]

        expected_bert_tokens = [
            [CLS, "john", "johan", "##son", "'", "##s",  "house", SEP],
            [CLS, "who", "was", "jim", "henson", "?", SEP]
        ]
        expected_orig_to_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]
        expected_bert_labels = [
            [WORDPIECE, "B-PER", "I-PER", WORDPIECE, "I-PER", WORDPIECE, "O", WORDPIECE],
            [WORDPIECE, "O", "O", "B-PER", "I-PER", "O", WORDPIECE]
        ]

        actual_bert_tokens, actual_orig_to_tok_map, actual_bert_labels = \
            bert_utils.wordpiece_tokenize_sents(tokens=tokens,
                                                tokenizer=bert_tokenizer,
                                                labels=labels)

        assert expected_bert_tokens == actual_bert_tokens
        assert expected_orig_to_tok_map == actual_orig_to_tok_map
        assert expected_bert_labels == actual_bert_labels

    def test_index_pad_mask_bert_tokens(self, bert_tokenizer):
        """Asserts that `bert_utils.index_pad_mask_bert_tokens()` returns the expected values for a
        simple input when input argument `labels` is None.
        """
        bert_tokens = [
            [CLS, "john", "johan", "##son", "'", "##s",  "house", SEP],
            [CLS, "who", "was", "jim", "henson", "?", SEP]
        ]
        orig_to_tok_map = [
            [1, 2, 4, 6],
            [1, 2, 3, 4, 5]
        ]

        expected_orig_to_tok_map = torch.as_tensor(
            [tm + [TOK_MAP_PAD] * (constants.MAX_SENT_LEN - len(tm)) for tm in orig_to_tok_map]
        )
        expected_attention_masks = torch.as_tensor([
            [1.] * len(bert_tokens[0]) + [0.] * (constants.MAX_SENT_LEN - len(bert_tokens[0])),
            [1.] * len(bert_tokens[1]) + [0.] * (constants.MAX_SENT_LEN - len(bert_tokens[1])),
        ])

        actual_indexed_tokens, actual_orig_to_tok_map, actual_attention_masks = \
            bert_utils.index_pad_mask_bert_tokens(tokens=bert_tokens,
                                                  orig_to_tok_map=orig_to_tok_map,
                                                  tokenizer=bert_tokenizer)

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, constants.MAX_SENT_LEN)
        assert torch.equal(actual_orig_to_tok_map, expected_orig_to_tok_map)
        assert torch.equal(expected_attention_masks, actual_attention_masks)

    def test_index_pad_mask_bert_tokens_labels(self, bert_tokenizer):
        """Asserts that `bert_utils.index_pad_mask_bert_tokens()` returns the expected values for a
        simple input when input argument `labels` is not None.
        """
        tokens = [
            ["john", "johanson", "'s",  "house"],
            ["who", "was", "jim", "henson",  "?"]
        ]
        labels = [
            ["B-PER", "I-PER", "I-PER",  "O"],
            ["O", "O", "B-PER", "I-PER",  "O"]
        ]
        bert_tokens, orig_to_tok_map, bert_labels = \
            bert_utils.wordpiece_tokenize_sents(tokens=tokens,
                                                tokenizer=bert_tokenizer,
                                                labels=labels)

        expected_orig_to_tok_map = torch.as_tensor(
            [tm + [TOK_MAP_PAD] * (constants.MAX_SENT_LEN - len(tm)) for tm in orig_to_tok_map]
        )
        expected_attention_masks = torch.as_tensor([
            [1.] * len(bert_tokens[0]) + [0.] * (constants.MAX_SENT_LEN - len(bert_tokens[0])),
            [1.] * len(bert_tokens[1]) + [0.] * (constants.MAX_SENT_LEN - len(bert_tokens[1])),
        ])
        # Create a padded sequece of labels mapped to their indices
        tag_to_idx = {
            PAD: 0,
            'O': 1,
            'B-PER': 2,
            'I-PER': 3,
            WORDPIECE: 4
        }
        expected_indexed_labels = torch.tensor(
            [[tag_to_idx[lab] for lab in sent] + [0] * (constants.MAX_SENT_LEN - len(sent))
             for sent in bert_labels]
        )

        (actual_indexed_tokens, actual_orig_to_tok_map, actual_attention_masks,
         actual_indexed_labels) = \
            bert_utils.index_pad_mask_bert_tokens(tokens=bert_tokens,
                                                  orig_to_tok_map=orig_to_tok_map,
                                                  tokenizer=bert_tokenizer,
                                                  labels=bert_labels,
                                                  tag_to_idx=tag_to_idx)

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, constants.MAX_SENT_LEN)
        assert torch.equal(actual_orig_to_tok_map, expected_orig_to_tok_map)
        assert torch.equal(expected_attention_masks, actual_attention_masks)
        assert torch.equal(expected_indexed_labels, actual_indexed_labels)

    def test_get_dataloader_for_bert_conll2003(self, conll2003datasetreader_load, bert_tokenizer):
        """Asserts that `bert_utils.get_dataloader_for_bert()` returns the expected values for a
        a given `Dataset` object.
        """
        processed_dataset = \
            bert_utils.process_dataset_for_bert(conll2003datasetreader_load, bert_tokenizer)
        dataloaders = bert_utils.get_dataloader_for_bert(processed_dataset, batch_size=32)

        for partition, dataloader in zip(constants.PARTITIONS, dataloaders.values()):
            # Get tensors that are loaded into DataLoader
            expected_input_ids, expected_attention_mask = processed_dataset[f'x_{partition}']
            expected_labels = processed_dataset[f'y_{partition}']
            expected_orig_to_tok_map = processed_dataset[f'orig_to_tok_map_{partition}']
            expected_model_idx = 0

            assert torch.equal(expected_input_ids, dataloader.dataset.input_ids)
            assert torch.equal(expected_attention_mask, dataloader.dataset.attention_mask)
            assert torch.equal(expected_labels, dataloader.dataset.labels)
            assert torch.equal(expected_orig_to_tok_map, dataloader.dataset.orig_to_tok_map)
            assert expected_model_idx == dataloader.dataset.model_idx

            assert dataloader.batch_size == 32

            if partition == 'train':
                assert isinstance(dataloader.sampler, RandomSampler)
            else:
                assert isinstance(dataloader.sampler, SequentialSampler)

    def test_get_dataloader_for_bert_conll2004(self, conll2004datasetreader_load, bert_tokenizer):
        """Asserts that `bert_utils.get_dataloader_for_bert()` returns the expected values for a
        a given `Dataset` object.
        """
        processed_dataset = \
            bert_utils.process_dataset_for_bert(conll2004datasetreader_load, bert_tokenizer)
        dataloaders = bert_utils.get_dataloader_for_bert(processed_dataset, batch_size=32)

        for partition, dataloader in zip(constants.PARTITIONS, dataloaders.values()):
            # Get tensors that are loaded into DataLoader
            expected_input_ids, expected_attention_mask = processed_dataset[f'x_{partition}']
            expected_labels = processed_dataset[f'y_{partition}']
            expected_orig_to_tok_map = processed_dataset[f'orig_to_tok_map_{partition}']
            expected_model_idx = 0

            assert torch.equal(expected_input_ids, dataloader.dataset.input_ids)
            assert torch.equal(expected_attention_mask, dataloader.dataset.attention_mask)
            assert torch.equal(expected_labels, dataloader.dataset.labels)
            assert torch.equal(expected_orig_to_tok_map, dataloader.dataset.orig_to_tok_map)
            assert expected_model_idx == dataloader.dataset.model_idx

            assert dataloader.batch_size == 32

            if partition == 'train':
                assert isinstance(dataloader.sampler, RandomSampler)
            else:
                assert isinstance(dataloader.sampler, SequentialSampler)

    def test_prepare_optimizers(self, bert_for_ner_model_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNER.prepare_optimizers()` for a single-task model.
        """
        actual = bert_for_ner_model_specify.prepare_optimizers()

        assert all(isinstance(opt, BertAdam) for opt in actual)

    def test_prepare_optimizers_mt(self, mt_bert_for_ner_model_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `BertForNER.prepare_optimizers()` for a multi-task model.
        """
        actual = mt_bert_for_ner_model_specify.prepare_optimizers()

        assert all(isinstance(opt, BertAdam) for opt in actual)
