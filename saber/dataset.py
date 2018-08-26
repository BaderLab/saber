"""Contains the Dataset class, which handles the loading and storage of datasets.
"""
from glob import glob
from itertools import chain
import logging
import os

from . import constants
from .preprocessor import Preprocessor
from .utils import model_utils

class Dataset(object):
    """A class for handling datasets.

    Args:
        filepath (str): path to directory containing a CoNLL formatted dataset
        sep (str): Seperator used for dataset at 'filepath'. Defaults to '\t'.
        replace_rare_tokens (bool): True if rare tokens should be replaced with a special unknown
            token. Threshold for considering tokens rare can be found at constants.NUM_RARE.
    """
    def __init__(self, filepath, sep='\t', replace_rare_tokens=True):
        self.log = logging.getLogger(__name__)
        # collect filepaths for each partition in a dictionary
        self.partition_filepaths = self.get_filepaths(filepath)
        # column delimiter in CONLL dataset
        self.sep = sep
        self.replace_rare_tokens = replace_rare_tokens

        # word and tag seq from dataset, a dict of dict containing partition: type key value pairs.
        self.type_seq = {'train': {'word': None, 'char': None, 'tag': None},
                         'valid': {'word': None, 'char': None, 'tag': None},
                         'test': {'word': None, 'char': None, 'tag': None},}

        # word, character, and tag types from dataset
        self.types = {'word': None, 'char': None, 'tag': None}
        # mappings of type: index for word, character and tag types
        self.type_to_idx = {'word': None, 'char': None, 'tag': None}
        # inverse mapping from indices to tag types
        self.idx_to_tag = None
        # same as type_seq, but all types mapped to int idx according to self.type_to_idx
        self.idx_seq = {'train': {'word': None, 'char': None, 'tag': None},
                        'valid': {'word': None, 'char': None, 'tag': None},
                        'test': {'word': None, 'char': None, 'tag': None},}

    def get_filepaths(self, filepath):
        """Collects 'train' (and, if they exist) 'valid'/'test' parition filepaths from `filepath`.

        Looks for 'train', 'valid' and 'test' partitions at 'filepath/train.*', 'filepath/valid.*'
        and 'filepath/test.*' respectively. Train set must be provided, valid and test sets are
        optional.

        Args:
            filepath (str): path to dataset (corpus)

        Returns:
            a dictionary with keys 'train', and optionally, 'valid' and 'test' containing the
            filepaths to the train, valid and test paritions of the dataset at 'filepath'.

        Raises:
            ValueError when no file at filepath/train.* is found.
        """
        partition_filepaths = {}
        # search for partition filepaths
        train_partition = glob(os.path.join(filepath, constants.TRAIN_FILE))
        valid_partition = glob(os.path.join(filepath, constants.VALID_FILE))
        test_partition = glob(os.path.join(filepath, constants.TEST_FILE))

        # must supply a train file
        if not train_partition:
            err_msg = "Must supply at least one file, train.* at {}".format(filepath)
            self.log.error('ValueError %s', err_msg)
            raise ValueError(err_msg)
        partition_filepaths['train'] = train_partition[0]

        # optionally, valid and test files can be supplied
        if valid_partition:
            partition_filepaths['valid'] = valid_partition[0]
        if test_partition:
            partition_filepaths['test'] = test_partition[0]

        return partition_filepaths

    def load_dataset(self, type_to_idx=None):
        """Coordinates loading of given data set at self.filepath.

        For a given dataset in CoNLL format at filepath, cordinates the loading of data and updates
        instance attributes. Expects self.filepath to be a directory containing a single file,
        train.* and optionally two additional files, valid.* and test.*. If this is a type_to_idx
        must be provided.

        Args:
            type_to_idx (dict): optional, a mapping of word types to indices
                (at type_to_idx['word']) and a mapping of char types to indices
                (at type_to_idx['char']) shared between datasets.
        """
        tag_map = constants.INITIAL_MAPPING['tag']
        # if None, not transfer learning and loading a single dataset
        if type_to_idx is None:
            self.load_data_and_labels()
            self.get_types()

            # generate type to index mappings
            self.type_to_idx['word'] = Preprocessor.type_to_idx(self.types['word'],
                                                                constants.INITIAL_MAPPING['word'])
            self.type_to_idx['char'] = Preprocessor.type_to_idx(self.types['char'],
                                                                constants.INITIAL_MAPPING['word'])
        # if not None, transfer learning or loading a compound dataset or both
        else:
            self.type_to_idx.update(type_to_idx)
            # if 'tag' exists, we are transfer learning, use mapping as starting point
            if 'tag' in type_to_idx:
                tag_map = type_to_idx['tag']

        # tag to idx maps are not shared in compound datasets, and are extending during TL
        self.type_to_idx['tag'] = Preprocessor.type_to_idx(self.types['tag'], tag_map)

        # create reverse mapping of indices to tags, save computation downstream
        self.idx_to_tag = {v: k for k, v in self.type_to_idx['tag'].items()}
        # use type to idx mapping to get final representation for training
        self.get_idx_seq()

    def load_data_and_labels(self):
        """Loads CoNLL format data set given at self.trainset_filepath.

        Updates self.word_seq and self.tag_seq each with a numpy array of lists containg the word
        and tag sequence respectively. Expects filepath to be a directory containing one file,
        train.*. The file format is tab-separated values. A blank line is required at the end of
        each sentence.

        Example dataset:
        '''
        The	O
        transcription	O
        of	O
        most	O
        RP	B-PRGE
        genes	I-PRGE
        ...	O
        '''
        """
        for partition, filepath in self.partition_filepaths.items():
            # accumulate sequences across a single partition
            word_seq, tag_seq = [], []

            with open(filepath, 'r') as ds:
                # accumulators per sentence
                words, tags = [], []

                for line in ds:
                    line = line.rstrip() # right strip

                    # accumulate each sequence
                    if not line or line.startswith('-DOCSTART-'):
                        if words:
                            word_seq.append(words)
                            tag_seq.append(tags)

                            words, tags = [], []
                    # accumulate each word/tag in a sequence
                    else:
                        # ignore comment lines
                        if not line.startswith('# '):
                            word, tag = line.split(self.sep)
                            words.append(word)
                            tags.append(tag)

                # in order to collect last sentence in the file
                word_seq.append(words)
                tag_seq.append(tags)

            # replace all rare tokens with special unknown token
            if self.replace_rare_tokens:
                word_seq = Preprocessor.replace_rare_tokens(word_seq)

            self.type_seq[partition] = {'word': word_seq, 'tag': tag_seq}

    def get_types(self):
        """Returns word types, char types and tag types.

        Returns the word types, character types and tag types for the current
        loaded dataset. All types are shared across all partitions, that is,
        word, char and tag types are collected from train, valid, and test
        partitions (if provided),

        Returns:
            a three-tuple containing the word types, chararcter types and
            tag types.
        """
        # Word types
        # iterate over all partitions which have a not None value at type_seq[partition]['word'],
        # that is, all partitions present in the dataset
        word_types = [s for p in self.type_seq.values() if p['word'] is not None for s in p['word']]
        # flatten list of word types, and convert to set to get unqiue elements
        word_types = list(set(chain.from_iterable(word_types)))

        # Char types
        char_types = []
        for word in word_types:
            char_types.extend(list(word))
        char_types = list(set(char_types))

        # Tag types
        tag_types = [s for p in self.type_seq.values() if p['tag'] is not None for s in p['tag']]
        tag_types = list(set(chain.from_iterable(tag_types)))

        self.types['word'] = word_types
        self.types['char'] = char_types
        self.types['tag'] = tag_types

    def get_idx_seq(self):
        """Get the index sequence for 'word', 'char', and 'tag' types for all paritions.

        For paritions 'train' and (optionally) 'valid' and 'test', maps all 'word', 'char', and
        'tag' types to a unique integer index. Returns a dictionary of dictionaries, where the
        first dictionary contains keys for each parition ('train', 'valid', 'test') and the
        nested dictionaries contains keys for each type ('word', 'char', 'tag').

        Returns:
            dictionary of dictionaries, containing the index sequences for 'word', 'char', and 'tag'
            types for all paritions 'train', 'valid' and 'test'.
        """
        for partition in self.partition_filepaths:
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

            self.idx_seq[partition]['tag'] = model_utils.idx_seq_to_categorical(
                self.idx_seq[partition]['tag'], num_classes=len(self.type_to_idx['tag'])
            )
