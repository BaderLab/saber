"""Contains the Dataset class, which handles the loading and storage of datasets.
"""
import logging
import os
from itertools import chain

from keras.utils import to_categorical
from nltk.corpus.reader.conll import ConllCorpusReader

from . import constants
from .preprocessor import Preprocessor
from .utils import data_utils, generic_utils

LOGGER = logging.getLogger(__name__)


class Dataset(object):
    """A class for handling datasets. Expects datasets to be in tab-seperated CoNLL format, where
    each line contains a token and its tag (seperated by a tab) and each sentence is seperated
    by a blank line.

    Example corpus:
    '''
    The	O
    transcription	O
    of	O
    most	O
    RP	B-PRGE
    genes	I-PRGE
    ...
    '''

    Args:
        dataset_folder (str): Path to directory containing CoNLL formatted dataset(s).
        replace_rare_tokens (bool): True if rare tokens should be replaced with a special unknown
            token. Threshold for considering tokens rare can be found at `saber.constants.NUM_RARE`.
    """
    def __init__(self, dataset_folder=None, replace_rare_tokens=False, **kwargs):
        self.dataset_folder = dataset_folder
        # don't load corpus unless `dataset_folder` was passed on object construction
        if self.dataset_folder is not None:
            self.dataset_folder = data_utils.get_filepaths(dataset_folder)
            self.conll_parser = ConllCorpusReader(dataset_folder, '.conll', ('words', 'pos'))

        self.replace_rare_tokens = replace_rare_tokens

        # word, character and tag sequences from dataset (per partition)
        self.type_seq = {'train': None, 'valid': None, 'test': None}
        # mappings of word, characters, and tag types to unique integer IDs
        self.type_to_idx = {'word': None, 'char': None, 'tag': None}
        # reverse mapping of unique integer IDs to tag types
        self.idx_to_tag = None
        # same as type_seq but all words, characters and tags have been mapped to unique integer IDs
        self.idx_seq = {'train': None, 'valid': None, 'test': None}

        for key, value in kwargs.items():
            setattr(self, key, value)

    def load(self):
        """Coordinates the loading of a given data set at `self.dataset_folder`.

        For a given dataset in CoNLL format at `self.dataset_folder`, coordinates the loading of
        data and updates the appropriate instance attributes. Expects `self.dataset_folder` to be a
        directory containing a single file, `train.*` and optionally two additional files, `valid.*`
        and `test.*`.

        Raises:
            ValueError if `self.dataset_folder` is None.
        """
        if self.dataset_folder is None:
            err_msg = ("`Dataset.dataset_folder` is None. Must be provided before call to "
                       "`Dataset.load()`")
            LOGGER.error('ValueError %s', err_msg)
            raise ValueError(err_msg)

        # unique words, chars and tags from CoNLL formatted dataset
        types = self._get_types()
        # map each word, char, and tag type to a unique integer
        self._get_idx_maps(types)
        # get word, char, and tag sequences from CoNLL formatted dataset
        self._get_type_seq()
        # get final representation used for training
        self.get_idx_seq()
        # useful during prediction / annotation
        self.get_idx_to_tag()

    def _get_types(self):
        """Collects the sets of all words, characters and tags in a CoNLL formatted dataset.

        For the CoNLL formatted dataset given at `self.dataset_folder`, updates `self.types` with
        the sets of all words (word types), characters (character types) and tags (tag types). All
        types are shared across all partitions, that is, word, char and tag types are collected from
        the train and, if provided, valid/test partitions found at `self.dataset_folder/train.*`,
        `self.dataset_folder/valid.*` and `self.dataset_folder/test.*`.

        Returns:
            A dictionary with keys 'word', 'char', 'tag' containing lists of unique words,
            characters and tags in the CoNLL formatted dataset at `self.dataset_folder`.
        """
        types = {'word': {constants.PAD, constants.UNK},
                 'char': {constants.PAD, constants.UNK},
                 'tag': {constants.PAD},
                 }

        for _, filepath in self.dataset_folder.items():
            if filepath is not None:
                conll_file = os.path.basename(filepath)  # get name of conll file

                words = self.conll_parser.words(conll_file)
                types['word'].update(words)
                types['char'].update(chain.from_iterable([list(w) for w in words]))
                types['tag'].update({tag[-1] for tag in self.conll_parser.tagged_words(conll_file)})

        types['word'] = list(types['word'])
        types['char'] = list(types['char'])
        types['tag'] = list(types['tag'])

        return types

    def _get_type_seq(self):
        """Loads sequence data from a CoNLL format data set given at `self.dataset_folder`.

        For the CoNLL formatted dataset given at `self.dataset_folder`, updates `self.type_seq` with
        lists containing the word, character and tag sequences for the train and, if provided,
        valid/test partitions found at `self.dataset_folder/train.*`, `self.dataset_folder/valid.*`
        and `self.dataset_folder/test.*`.
        """
        for partition, filepath in self.dataset_folder.items():
            if filepath is not None:
                conll_file = os.path.basename(filepath)  # get name of conll file

                # collect sequence data
                sents = list(self.conll_parser.sents(conll_file))
                tagged_sents = list(self.conll_parser.tagged_sents(conll_file))

                word_seq = (Preprocessor.replace_rare_tokens(sents)
                            if self.replace_rare_tokens else sents)
                char_seq = [[[c for c in w] for w in s] for s in sents]
                tag_seq = [[t[-1] for t in s] for s in tagged_sents]

                # update the class attributes
                self.type_seq[partition] = {'word': word_seq, 'char': char_seq, 'tag': tag_seq}

    def _get_idx_maps(self, types, initial_mapping=None):
        """Updates `self.type_to_idx` with mappings from word, char and tag types to unique IDs.

        Args:
            types (dict): A dictionary of lists containing unique types (word, char, and tag) loaded
                from a CoNLL formatted dataset.
            initial_mapping (dict): A dictionary mapping types to unique integer IDs. If provided,
                the type to index mapping, `self.type_to_idx`, will update this initial mapping.
                If None, `constants.INITIAL_MAPPING` is used.
        """
        initial_mapping = constants.INITIAL_MAPPING if initial_mapping is None else initial_mapping
        # generate type to index mappings
        self.type_to_idx['word'] = Preprocessor.type_to_idx(types['word'], initial_mapping['word'])
        self.type_to_idx['char'] = Preprocessor.type_to_idx(types['char'], initial_mapping['word'])
        self.type_to_idx['tag'] = Preprocessor.type_to_idx(types['tag'], initial_mapping['tag'])

    def get_idx_seq(self):
        """Updates `self.idx_seq` with the final representation of the data used for training.

        Updates `self.idx_seq` with numpy arrays, by using `self.type_to_idx` to map all elements
        in `self.type_seq` to their corresponding integer IDs, for the train and, if provided,
        valid/test partitions found at `self.dataset_folder/train.*`, `self.dataset_folder/valid.*`
        and `self.dataset_folder/test.*`.
        """
        for partition, filepath in self.dataset_folder.items():
            if filepath is not None:
                self.idx_seq[partition] = {
                    'word': Preprocessor.get_type_idx_sequence(self.type_seq[partition]['word'],
                                                               self.type_to_idx['word'],
                                                               type_='word'),
                    'char': Preprocessor.get_type_idx_sequence(self.type_seq[partition]['word'],
                                                               self.type_to_idx['char'],
                                                               type_='char'),
                    'tag': Preprocessor.get_type_idx_sequence(self.type_seq[partition]['tag'],
                                                              self.type_to_idx['tag'],
                                                              type_='tag'),
                }
                # one-hot encode our targets
                self.idx_seq[partition]['tag'] = to_categorical(self.idx_seq[partition]['tag'])

    def get_idx_to_tag(self):
        """Updates `self.idx_to_tag` with a reverse mapping of `self.type_to_idx['tag']`.

        Updates `self.idx_to_tag` with a dictionary that maps all values in
        `self.type_to_idx['tag']` to keys in `self.type_to_idx['tag']`. This mapping is useful
        for decoding the predictions made by a model back to the corresponding tag sequence.
        """
        self.idx_to_tag = generic_utils.reverse_dict(self.type_to_idx['tag'])
