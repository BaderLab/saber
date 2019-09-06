"""Test suite for the `bert_utils` module (saber.utils.bert_utils).
"""
import torch
from pytorch_transformers.optimization import AdamW
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler

from ..constants import CLS
from ..constants import MAX_SENT_LEN
from ..constants import PAD
from ..constants import SEP
from ..constants import TOK_MAP_PAD
from ..constants import WORDPIECE
from ..utils import bert_utils

# Value reused across multiple tests
tokens = [["john", "johanson", "'s",  "house"], ["who", "was", "jim", "henson",  "?"]]
bert_tokens = [
    [CLS, "john", "johan", "##son", "'", "##s",  "house", SEP],
    [CLS, "who", "was", "jim", "henson", "?", SEP]
]

labels = [["B-PER", "I-PER", "I-PER",  "O"], ["O", "O", "B-PER", "I-PER",  "O"]]
bert_labels = [
    [WORDPIECE, "B-PER", "I-PER", WORDPIECE, "I-PER", WORDPIECE, "O", WORDPIECE],
    [WORDPIECE, "O", "O", "B-PER", "I-PER", "O", WORDPIECE]
]

orig_to_tok_map = [[1, 2, 4, 6], [1, 2, 3, 4, 5]]

tag_to_idx = {
    PAD: 0,
    'O': 1,
    'B-PER': 2,
    'I-PER': 3,
    WORDPIECE: 4
}

attention_mask = torch.as_tensor([
    [1.] * len(bert_tokens[0]) + [0.] * (MAX_SENT_LEN - len(bert_tokens[0])),
    [1.] * len(bert_tokens[1]) + [0.] * (MAX_SENT_LEN - len(bert_tokens[1])),
])


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
            assert actual[partition]['x'][0].size() == (2, MAX_SENT_LEN)
            assert actual[partition]['x'][1].size() == (2, MAX_SENT_LEN)
            assert actual[partition]['y'].size() == (2, MAX_SENT_LEN)
            assert actual[partition]['orig_to_tok_map'].size() == (2, MAX_SENT_LEN)
            # This is a CoNLL2003 formatted dataset so there should be no rel labels
            assert 'rel_labels' not in actual[partition]

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
            assert actual[partition]['x'][0].size() == (3, MAX_SENT_LEN)
            assert actual[partition]['x'][1].size() == (3, MAX_SENT_LEN)
            assert actual[partition]['y'].size() == (3, MAX_SENT_LEN)
            assert actual[partition]['orig_to_tok_map'].size() == (3, MAX_SENT_LEN)
            assert actual[partition]['rel_labels'] == \
                conll2004datasetreader_load.idx_seq[partition]['rel']

    def test_wordpiece_tokenize_sents(self, bert_tokenizer):
        """Asserts that `bert_utils.wordpiece_tokenize_sents()` returns the expected values for a
        simple input when input argument `labels` is None.
        """
        expected = (bert_tokens, orig_to_tok_map)

        actual = bert_utils.wordpiece_tokenize_sents(tokens, tokenizer=bert_tokenizer)

        assert expected == actual

    def test_wordpiece_tokenize_sents_labels(self, bert_tokenizer):
        """Asserts that `bert_utils.wordpiece_tokenize_sents()` returns the expected values for a
        simple input when input argument `labels` is not None.
        """
        expected = (bert_tokens, orig_to_tok_map, bert_labels)

        actual = \
            bert_utils.wordpiece_tokenize_sents(tokens, tokenizer=bert_tokenizer, labels=labels)

        assert expected == actual

    def test_index_pad_mask_bert_tokens(self, bert_tokenizer):
        """Asserts that `bert_utils.index_pad_mask_bert_tokens()` returns the expected values for a
        simple input when input argument `labels` is None.
        """
        actual_indexed_tokens, actual_orig_to_tok_map, actual_attention_mask = \
            bert_utils.index_pad_mask_bert_tokens(bert_tokens,
                                                  orig_to_tok_map=orig_to_tok_map,
                                                  tokenizer=bert_tokenizer,
                                                  maxlen=MAX_SENT_LEN)

        expected_orig_to_tok_map = torch.as_tensor(
            [tm + [TOK_MAP_PAD] * (MAX_SENT_LEN - len(tm)) for tm in orig_to_tok_map]
        )

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, MAX_SENT_LEN)
        assert torch.equal(expected_orig_to_tok_map, actual_orig_to_tok_map)
        assert torch.equal(attention_mask, actual_attention_mask)

    def test_index_pad_mask_bert_tokens_labels(self, bert_tokenizer):
        """Asserts that `bert_utils.index_pad_mask_bert_tokens()` returns the expected values for a
        simple input when input argument `labels` is not None.
        """
        (actual_indexed_tokens, actual_orig_to_tok_map, actual_attention_mask,
         actual_indexed_labels) = \
            bert_utils.index_pad_mask_bert_tokens(tokens=bert_tokens,
                                                  orig_to_tok_map=orig_to_tok_map,
                                                  tokenizer=bert_tokenizer,
                                                  maxlen=MAX_SENT_LEN,
                                                  labels=bert_labels,
                                                  tag_to_idx=tag_to_idx)

        expected_orig_to_tok_map = torch.as_tensor(
            [tm + [TOK_MAP_PAD] * (MAX_SENT_LEN - len(tm)) for tm in orig_to_tok_map]
        )
        expected_indexed_labels = torch.tensor(
            [[tag_to_idx[lab] for lab in sent] + [0] * (MAX_SENT_LEN - len(sent))
             for sent in bert_labels]
        )

        # Just check for shape, as token indicies will depend on specific BERT model used
        assert actual_indexed_tokens.shape == (2, MAX_SENT_LEN)
        assert torch.equal(expected_orig_to_tok_map, actual_orig_to_tok_map)
        assert torch.equal(attention_mask, actual_attention_mask)
        assert torch.equal(expected_indexed_labels, actual_indexed_labels)

    def test_get_dataloader_for_bert_conll2003(self, conll2003datasetreader_load, bert_tokenizer):
        """Asserts that `bert_utils.get_dataloader_for_bert()` returns the expected values for a
        a given `Dataset` object.
        """
        processed_dataset = \
            bert_utils.process_dataset_for_bert(conll2003datasetreader_load, bert_tokenizer)
        dataloaders = bert_utils.get_dataloader_for_bert(processed_dataset, batch_size=32)

        for partition, dataloader in dataloaders.items():
            # Get tensors that are loaded into DataLoader
            expected_input_ids, expected_attention_mask = processed_dataset[partition]['x']
            expected_labels = processed_dataset[partition]['y']
            expected_orig_to_tok_map = processed_dataset[partition]['orig_to_tok_map']
            expected_model_idx = -1

            assert torch.equal(expected_input_ids, dataloader.dataset.input_ids)
            assert torch.equal(expected_attention_mask, dataloader.dataset.attention_mask)
            assert torch.equal(expected_labels, dataloader.dataset.labels)
            assert torch.equal(expected_orig_to_tok_map, dataloader.dataset.orig_to_tok_map)
            assert expected_model_idx == dataloader.dataset.model_idx

            if partition == 'train':
                assert dataloader.batch_size == 32
                assert isinstance(dataloader.sampler, RandomSampler)
            else:
                assert isinstance(dataloader.sampler, SequentialSampler)
                assert dataloader.batch_size == 32 * 4

    def test_get_dataloader_for_bert_conll2004(self, conll2004datasetreader_load, bert_tokenizer):
        """Asserts that `bert_utils.get_dataloader_for_bert()` returns the expected values for a
        a given `Dataset` object.
        """
        processed_dataset = \
            bert_utils.process_dataset_for_bert(conll2004datasetreader_load, bert_tokenizer)
        dataloaders = bert_utils.get_dataloader_for_bert(processed_dataset, batch_size=32)

        for partition, dataloader in dataloaders.items():
            # Get tensors that are loaded into DataLoader
            expected_input_ids, expected_attention_mask = processed_dataset[partition]['x']
            expected_labels = processed_dataset[partition]['y']
            expected_orig_to_tok_map = processed_dataset[partition]['orig_to_tok_map']
            expected_model_idx = -1

            assert torch.equal(expected_input_ids, dataloader.dataset.input_ids)
            assert torch.equal(expected_attention_mask, dataloader.dataset.attention_mask)
            assert torch.equal(expected_labels, dataloader.dataset.labels)
            assert torch.equal(expected_orig_to_tok_map, dataloader.dataset.orig_to_tok_map)
            assert expected_model_idx == dataloader.dataset.model_idx

            if partition == 'train':
                assert dataloader.batch_size == 32
                assert isinstance(dataloader.sampler, RandomSampler)
            else:
                assert isinstance(dataloader.sampler, SequentialSampler)
                assert dataloader.batch_size == 32 * 4

    def test_get_bert_optimizer(self, bert_for_ner_specify):
        """Asserts that the returned optimizer object is as expected after call to
        `bert_utils.get_bert_optimizer()` for a single-task model.
        """
        model, config = bert_for_ner_specify.model, bert_for_ner_specify.config

        actual = bert_utils.get_bert_optimizer(config, model)

        assert isinstance(actual, AdamW)
