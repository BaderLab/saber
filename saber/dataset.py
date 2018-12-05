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
        directory (str): Path to directory containing CoNLL formatted dataset(s).
        replace_rare_tokens (bool): True if rare tokens should be replaced with a special unknown
            token. Threshold for considering tokens rare can be found at `saber.constants.NUM_RARE`.
    """
    def __init__(self, directory=None, replace_rare_tokens=True, **kwargs):
        self.directory = directory
        # don't load corpus unless directory was passed on object construction
        if self.directory is not None:
            self.directory = data_utils.get_filepaths(directory)
            self.conll_parser = ConllCorpusReader(directory, '.conll', ('words', 'pos'))

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
        """Coordinates the loading of a given data set at `self.directory`.

        For a given dataset in CoNLL format at `self.directory`, coordinates the loading of data and
        updates the appropriate instance attributes. Expects `self.directory` to be a directory
        containing a single file, `train.*` and optionally two additional files, `valid.*` and
        `test.*`.

        Raises:
            ValueError if `self.directory` is None.
        """
        if self.directory is None:
            err_msg = "`Dataset.directory` is None; must be provided before call to `Dataset.load`"
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
        self.idx_to_tag = generic_utils.reverse_dict(self.type_to_idx['tag'])

    def _get_types(self):
        """Collects the sets of all words, characters and tags in a CoNLL formatted dataset.

        For the CoNLL formatted dataset given at `self.directory`, updates `self.types` with the
        sets of all words (word types), characters (character types) and tags (tag types). All types
        are shared across all partitions, that is, word, char and tag types are collected from the
        train and, if provided, valid/test partitions found at `self.directory/train.*`,
        `self.directory/valid.*` and `self.directory/test.*`.
        """
        types = {'word': [constants.PAD, constants.UNK],
                 'char': [constants.PAD, constants.UNK],
                 'tag': [constants.PAD],
                }

        for _, filepath in self.directory.items():
            if filepath is not None:
                conll_file = os.path.basename(filepath)  # get name of conll file
                types['word'].extend(set(self.conll_parser.words(conll_file)))
                types['char'].extend(set(chain(*[list(w) for w in self.conll_parser.words(conll_file)])))
                types['tag'].extend(set([tag[-1] for tag in self.conll_parser.tagged_words(conll_file)]))

        # ensure that we have only unique types
        types['word'] = list(set(types['word']))
        types['char'] = list(set(types['char']))
        types['tag'] = list(set(types['tag']))

        return types

    def _get_type_seq(self):
        """Loads sequence data from a CoNLL format data set given at `self.directory`.

        For the CoNLL formatted dataset given at `self.directory`, updates `self.type_seq` with
        lists containing the word, character and tag sequences for the train and, if provided,
        valid/test partitions found at `self.directory/train.*`, `self.directory/valid.*` and
        `self.directory/test.*`.
        """
        for partition, filepath in self.directory.items():
            if filepath is not None:
                conll_file = os.path.basename(filepath) # get name of conll file

                # collect sequence data
                sents = list(self.conll_parser.sents(conll_file))
                tagged_sents = list(self.conll_parser.tagged_sents(conll_file))

                word_seq = Preprocessor.replace_rare_tokens(sents) if self.replace_rare_tokens else sents
                char_seq = [[[c for c in w] for w in s] for s in sents]
                tag_seq = [[t[-1] for t in s] for s in tagged_sents]

                # update the class attributes
                self.type_seq[partition] = {'word': word_seq, 'char': char_seq, 'tag': tag_seq}

    def _get_idx_maps(self, types, initial_mapping=None):
        """Updates `self.type_to_idx` with mappings from word, char and tag types to unique int IDs.
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
        valid/test partitions found at `self.directory/train.*`, `self.directory/valid.*` and
        `self.directory/test.*`.
        """
        for partition, filepath in self.directory.items():
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
