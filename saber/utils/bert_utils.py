import logging

import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert.optimization import BertAdam
from torch.utils import data

from saber import constants
from saber.constants import CLS
from saber.constants import SEP
from saber.constants import TOK_MAP_PAD
from saber.constants import WORDPIECE

LOGGER = logging.getLogger(__name__)

# TODO (johnmgiorgi): This should be handeled better. Maybe as a config argument.
FULL_FINETUNING = True


class BertDataset(data.Dataset):
    """A custom `torch.utils.data.Dataset` object for use with a BERT based model.
    """
    def __init__(self, input_ids, attention_mask, labels, orig_to_tok_map, model_idx):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.orig_to_tok_map = orig_to_tok_map
        self.model_idx = model_idx

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.input_ids)

    def __getitem__(self, index):
        """Generates one sample of data."""
        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]
        labels = self.labels[index]
        orig_to_tok_map = self.orig_to_tok_map[index]
        model_idx = self.model_idx

        return index, input_ids, attention_mask, labels, orig_to_tok_map, model_idx


def process_dataset_for_bert(dataset, tokenizer):
    """Process a Saber dataset to be compatible with a BERT-based model.

    Args:
        dataset (Dataset): A Dataset object for which `Dataset.load()` has been called.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.

    Returns
        A dictionary, keyed by '<partition_x>' and '<partition_y>' representing the processed
        input and target sequences for a given partition in `dataset`.
    """
    # Required for dataset to be compatible with BERTs WordPiece tokenization
    if WORDPIECE not in dataset.type_to_idx['ent']:
        dataset.type_to_idx['ent'][WORDPIECE] = len(dataset.type_to_idx['ent'])
        dataset.get_idx_to_tag()

    processed_dataset = {}

    for partition in constants.PARTITIONS:
        if dataset.type_seq[partition] is not None:
            # Tokenize pre-tokenized text using WordPiece tokenizer
            bert_tokens, orig_to_tok_map, bert_labels = \
                wordpiece_tokenize_sents(tokens=dataset.type_seq[partition]['word'],
                                         tokenizer=tokenizer,
                                         labels=dataset.type_seq[partition]['ent'])

            # Index, pad and mask BERT tokens
            indexed_tokens, orig_to_tok_map, attention_masks, indexed_labels = \
                index_pad_mask_bert_tokens(tokens=bert_tokens,
                                           orig_to_tok_map=orig_to_tok_map,
                                           tokenizer=tokenizer,
                                           labels=bert_labels,
                                           tag_to_idx=dataset.type_to_idx['ent'])

            processed_dataset[f'x_{partition}'] = [indexed_tokens, attention_masks]
            processed_dataset[f'y_{partition}'] = indexed_labels
            processed_dataset[f'orig_to_tok_map_{partition}'] = orig_to_tok_map
            # TODO (John): This is just a stand-in so we can get training on BioNLP ST 2019 ASAP.
            # The process for loading relation data will need to be re-thought.
            if 'rel' in dataset.idx_seq[partition]:
                processed_dataset[f'rel_labels_{partition}'] = dataset.idx_seq[partition]['rel']
        # TODO (John): Do we need to make these all None?
        else:
            processed_dataset[f'x_{partition}'] = None
            processed_dataset[f'y_{partition}'] = None
            processed_dataset[f'orig_to_tok_map_{partition}'] = None
            processed_dataset[f'rel_labels_{partition}'] = None

    return processed_dataset


# TODO (John): This does not work properly. The [SEP] tag can get dropped after padding.
def wordpiece_tokenize_sents(tokens, tokenizer, labels=None):
    """Tokenizes pre-tokenized text for use with a BERT-based model.

    Given some pre-tokenized text, represented as a list (sentences) of lists (tokens), tokenizies
    the text for use with a BERT-based model while deterministically maintaining an
    original-to-tokenized alignment. This is a near direct copy of the example given in the BERT
    GitHub repo (https://github.com/google-research/bert#tokenization) with additional code for
    mapping token-level labels.

    Args:
        tokens (list): A list of lists containing tokenized sentences.
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.
        labels (list): Optional, a list of lists containing token-level labels for a collection of
            sentences. Defaults to None.

    Returns:
        If `labels` is not `None`:
            A tuple of `bert_tokens`, `orig_to_tok_map`, `bert_labels`, representing tokens and
            labels that can be used to train a BERT model and a deterministc mapping of the elements
            in `bert_tokens` to `tokens`.
        If `labels` is `None`:
            A tuple of `bert_tokens`, and `orig_to_tok_map`, representing tokens that can be used to
            train a BERT model and a deterministc mapping of `bert_tokens` to `sents`.

    References:
     - https://github.com/google-research/bert#tokenization
    """
    bert_tokens = []
    orig_to_tok_map = []

    for sent in tokens:
        bert_tokens.append([CLS])
        orig_to_tok_map.append([])
        for orig_token in sent:
            orig_to_tok_map[-1].append(len(bert_tokens[-1]))
            bert_tokens[-1].extend(tokenizer.wordpiece_tokenizer.tokenize(orig_token))
        bert_tokens[-1].append(SEP)

    # If labels are provided, project them onto bert_tokens
    if labels is not None:
        bert_labels = []
        for bert_toks, labs, tok_map in zip(bert_tokens, labels, orig_to_tok_map):
            labs_iter = iter(labs)
            bert_labels.append([])
            for i, _ in enumerate(bert_toks):
                bert_labels[-1].extend([WORDPIECE if i not in tok_map else next(labs_iter)])

        return bert_tokens, orig_to_tok_map, bert_labels

    return bert_tokens, orig_to_tok_map


def index_pad_mask_bert_tokens(tokens, orig_to_tok_map, tokenizer, labels=None, tag_to_idx=None):
    """Convert `tokens` to indices, pad them, and generate the corresponding attention masks.

    Converts a list of tokenized sentences to indicies using the given `BertTokenizer`, pads this
    sequence and generates the corresponding attention masks.

    Args:
        tokens (list): A list of lists containing tokenized sentences.
        orig_to_tok_map (TODO).
        tokenizer (BertTokenizer): An object with methods for tokenizing text for input to BERT.
        labels (list): A list of lists containing token-level labels for a collection of sentences.
        tag_to_idx (dictionary): A dictionary mapping token-level tags/labels to unique integers.

    Returns:
        If `labels` is not `None`:
            A tuple of `torch.Tensor`'s: `indexed_tokens`, `attention_masks`, and `indexed_labels`
            that can be used as input to to train a BERT model. Note that if `labels` is not `None`,
            `tag_to_idx` must also be provided.
        If `labels` is `None`:
            A tuple of `torch.Tensor`'s: `indexed_tokens`, and `attention_masks`, representing
            tokens mapped to indices and corresponding attention masks that can be used as input to
            a BERT model.
    """
    # Convert sequences to indices and pad
    indexed_tokens = pad_sequences(
        sequences=[tokenizer.convert_tokens_to_ids(sent) for sent in tokens],
        maxlen=constants.MAX_SENT_LEN,
        dtype='long',
        padding='post',
        truncating='post',
        value=constants.PAD_VALUE)
    indexed_tokens = torch.as_tensor(indexed_tokens)

    orig_to_tok_map = pad_sequences(
        sequences=orig_to_tok_map,
        maxlen=constants.MAX_SENT_LEN,
        dtype='long',
        padding='post',
        truncating='post',
        value=TOK_MAP_PAD)
    orig_to_tok_map = torch.as_tensor(orig_to_tok_map)

    # Generate attention masks for pad values
    attention_masks = torch.as_tensor([[float(idx > 0) for idx in sent] for sent in indexed_tokens])

    if labels:
        indexed_labels = pad_sequences(
            sequences=[[tag_to_idx[lab] for lab in sent] for sent in labels],
            maxlen=constants.MAX_SENT_LEN,
            dtype='long',
            padding="post",
            truncating="post",
            value=constants.PAD_VALUE
        )
        indexed_labels = torch.as_tensor(indexed_labels)

        return indexed_tokens, orig_to_tok_map, attention_masks, indexed_labels

    return indexed_tokens, orig_to_tok_map, attention_masks


def get_dataloader_for_bert(processed_dataset, batch_size, model_idx=0):
    """Returns a `DataLoader` for the given `processed_dataset`,

    For the given `processed_dataset` (see `bert_utils.process_dataset_for_bert`) returns a
    dictionary of `DataLoader` objects, one per partition in `processed_dataset`.

    Args:
        processed_dataset (dict): A dictionary containing a dataset processed for use with a
            BERT-based model. See `bert_utils.process_dataset_for_bert`.
        attention_mask (numpy.ndarray): Numpy array or list containing the corresponding
            attention masks labels for `x`.
        batch_size (int): Batch size of the returned `DataLoader`'s.
        model_idx (int): Optional, an integer representing the model index the returned DataLoaders
            will correspond to. A tensor will be added to the `DataLoader`'s containing this
            integer, expanded with `torch.expand` to match the number of input examples. This is
            to be used in the case of multi-task models. Defaults to 0.

    Raises:
        ValueError if `data_partition` is not `train` or `eval`.

    Returns:
        A Torch DataLoader object for `x`, `y`, and `attention_masks`.
    """
    dataloaders = {}

    for partition in constants.PARTITIONS:
        if processed_dataset[f'x_{partition}'] is not None:
            x, attention_mask = processed_dataset[f'x_{partition}']
            y = processed_dataset[f'y_{partition}']
            orig_to_tok_map = processed_dataset[f'orig_to_tok_map_{partition}']

            dataset = BertDataset(x, attention_mask, y, orig_to_tok_map, model_idx)

            if partition == 'train':
                sampler = data.RandomSampler(dataset)
            else:
                sampler = data.SequentialSampler(dataset)

            dataloaders[f'dataloader_{partition}'] = data.DataLoader(dataset=dataset,
                                                                     sampler=sampler,
                                                                     batch_size=batch_size)
        else:
            dataloaders[f'dataloader_{partition}'] = None

    return dataloaders


def get_bert_optimizer(model, config):
    """Returns an Adam optimizer configured for optimization of a BERT model (`model`).

    Args:
        config (Config): Contains a set of harmonized arguments provided in a *.ini file and,
            optionally, from the command line.

    Returns:
        An initialized `BertAdam` optimizer for the training of a BERT model (`model`).
    """
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01,
             },
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             }
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,)

    return optimizer
