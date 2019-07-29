import logging
import os
from itertools import chain

import numpy as np
from nltk.corpus.reader.conll import ConllCorpusReader
from sklearn.utils.class_weight import compute_class_weight

from . import constants
from .constants import NEG
from .constants import PAD
from .constants import UNK
from .preprocessor import Preprocessor
from .utils import data_utils
from .utils import generic_utils

LOGGER = logging.getLogger(__name__)


class Dataset(object):
    """A base class which all dataset readers subclass.

    Args:
        dataset_folder (str): Path to directory containing CoNLL formatted dataset.
        replace_rare_tokens (bool): True if rare tokens should be replaced with a special unknown
            token. Threshold for considering tokens rare can be found at `saber.constants.NUM_RARE`.
    """
    def __init__(self, dataset_folder=None, replace_rare_tokens=False):
        self.dataset_folder = dataset_folder
        # Don't load corpus unless `dataset_folder` was passed on object construction
        if self.dataset_folder is not None:
            self.dataset_folder = data_utils.get_filepaths(dataset_folder)

        self.replace_rare_tokens = replace_rare_tokens

        # Word, character and tag sequences from dataset (per partition)
        self.type_seq = {'train': None, 'valid': None, 'test': None}
        # Mappings of word, characters, and tag types to unique integer IDs
        self.type_to_idx = {'word': None, 'char': None, 'ent': None, 'rel': None}
        # Reverse mapping of unique integer IDs to tag types
        self.idx_to_tag = {'ent': None, 'rel': None}
        # type_seq, but all words, characters and tags have been mapped to unique integer IDs
        self.idx_seq = {'train': None, 'valid': None, 'test': None}

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

        # Get word, char, and tag sequences from CoNLL formatted dataset
        self._get_type_seq()
        # Get unique words, chars and tags
        types = self._get_types()
        # Map each word, char, and tag type to a unique integer
        self._get_idx_maps(types)
        # Get final representation used for training
        self.get_idx_seq()
        # Useful during prediction / annotation
        self.get_idx_to_tag()

    def _get_types(self):
        """Collects the sets of all words, characters and tags in a CoNLL formatted dataset.

        For the CoNLL formatted dataset given at `self.dataset_folder`, returns a dictionary
        containing the sets of all words ('word'), characters ('char') and tags ('ent', 'rel'). All
        types are shared across all partitions, that is, word, char and tag types are collected from
        the train and, if provided, valid/test partitions found at `self.dataset_folder/train.*`,
        `self.dataset_folder/valid.*` and `self.dataset_folder/test.*`.

        Returns:
            A dictionary with keys 'word', 'char', 'ent' and 'rel' containing lists of unique words,
            characters and tags in the CoNLL formatted dataset at `self.dataset_folder`.
        """
        types = {'word': {PAD, UNK},
                 'char': {PAD, UNK},
                 'ent': {PAD},
                 'rel': {NEG},
                 }

        for partition, filepath in self.dataset_folder.items():
            if filepath is not None:

                words = tuple(set(chain(*self.type_seq[partition]['word'])))
                chars = tuple(set(chain(*[list(w) for w in words])))
                ents = tuple((set(chain(*self.type_seq[partition]['ent']))))

                if self.type_seq[partition]['rel'] is not None:
                    rels = tuple({t[-1] for s in self.type_seq[partition]['rel'] for t in s})
                    types['rel'].update(rels)

                types['word'].update(words)
                types['char'].update(chars)
                types['ent'].update(ents)

        types['word'] = list(types['word'])
        types['char'] = list(types['char'])
        types['ent'] = list(types['ent'])
        types['rel'] = list(types['rel'])

        return types

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
        # Generate type to index mappings
        self.type_to_idx['word'] = Preprocessor.type_to_idx(types['word'], initial_mapping['word'])
        self.type_to_idx['char'] = Preprocessor.type_to_idx(types['char'], initial_mapping['word'])
        self.type_to_idx['ent'] = Preprocessor.type_to_idx(types['ent'], initial_mapping['ent'])
        # Assume no relations to load if types['rel'] only contains the NEG class
        if types['rel'] != [NEG]:
            self.type_to_idx['rel'] = Preprocessor.type_to_idx(types['rel'], initial_mapping['rel'])

    # TODO (John): One-hot encoding should be an optional argument and work for both ents/rels.
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
                    'ent': Preprocessor.get_type_idx_sequence(self.type_seq[partition]['ent'],
                                                              self.type_to_idx['ent'],
                                                              type_='ent'),
                }

                if self.type_to_idx['rel'] is not None:
                    self.idx_seq[partition]['rel'] = \
                        Preprocessor.get_type_idx_sequence(self.type_seq[partition]['rel'],
                                                           self.type_to_idx['rel'],
                                                           type_='rel',
                                                           pad=False)

    def get_idx_to_tag(self):
        """Updates `self.idx_to_tag` with a reverse mapping of `self.type_to_idx['ent']` and,
        if .

        Updates `self.idx_to_tag` with a dictionary that maps all values in
        `self.type_to_idx['ent']` to keys in `self.type_to_idx['ent']`. This mapping is useful
        for decoding the predictions made by a model back to the corresponding tag sequence.
        """
        self.idx_to_tag['ent'] = generic_utils.reverse_dict(self.type_to_idx['ent'])
        if self.type_to_idx['rel'] is not None:
            self.idx_to_tag['rel'] = generic_utils.reverse_dict(self.type_to_idx['rel'])

    def compute_class_weight(self):
        """Returns the class weights for tag types 'ent' and if provided, 'rel'.

        For each tag type 'ent' (for entities) and 'rel' (for relations, if
        `self.type_to_idx['rel'] is not None`) returns a dictionary of class weight vectors which
        contains balanced class weights, given by `n_samples / (n_classes * np.bincount(y))`.
        """
        class_weights = {
            'ent': None,
            'rel': None,
        }

        ents = [e for s in self.idx_seq['train']['ent'] for e in s]
        class_weights['ent'] = compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(ents),
                                                    y=ents)
        # TODO (John): What to do with NEG value?
        if self.type_to_idx['rel'] is not None:
            rels = [r[-1] for s in self.idx_seq['train']['rel'] for r in s if r]
            class_weights['rel'] = compute_class_weight(class_weight='balanced',
                                                        classes=np.unique(rels),
                                                        y=rels)

        return class_weights


class CoNLL2003DatasetReader(Dataset):
    """A class for reading CoNLL2003 formatted datasets.

    Expects datasets to be in tab-seperated CoNLL 2003 format, where each line contains a token and
    its tag (seperated by a tab) and each sentence is seperated by a blank line.

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
        dataset_folder (str): Path to directory containing CoNLL formatted dataset.
        replace_rare_tokens (bool): True if rare tokens should be replaced with a special unknown
            token. Threshold for considering tokens rare can be found at `saber.constants.NUM_RARE`.
    """
    def __init__(self, dataset_folder=None, replace_rare_tokens=False):
        super(CoNLL2003DatasetReader, self).__init__(dataset_folder, replace_rare_tokens)

        if dataset_folder is not None:
            self.conll_parser = ConllCorpusReader(dataset_folder, '.conll', ('words', 'pos'))

    def _get_type_seq(self):
        """Loads sequence data from a CoNLL 2003 formated data set given at `self.dataset_folder`.

        For the CoNLL 2003 formatted dataset given at `self.dataset_folder`, updates
        `self.type_seq` with list of lists containing the word, character and tag sequences for the
        train and, if provided, valid/test partitions found at `self.dataset_folder/train.*`,
        `self.dataset_folder/valid.*` and `self.dataset_folder/test.*`.
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
                self.type_seq[partition] = {'word': word_seq,
                                            'char': char_seq,
                                            'ent': tag_seq,
                                            'rel': None
                                            }


class CoNLL2004DatasetReader(Dataset):
    """A class for reading CoNLL2004 formatted datasets.

    Expects datasets to be in tab-seperated
    CoNLL 2004 format. See: http://cogcomp.org/page/resource_view/43 for more information

    Example corpus:
    '''
    0	S-Var	0	O	O	Mutations	O	O	O
    0	O	1	O	O	in	O	O	O
    0	B-Gene	2	O	O	SHP	O	O	O
    0	I-Gene	3	O	O	-	O	O	O
    0	E-Gene	4	O	O	2	O	O	O
    0	S-Enzyme    5	O	O	phosphatase	O	O	O
    ...
    '''

    Args:
        dataset_folder (str): Path to directory containing CoNLL formatted dataset.
        replace_rare_tokens (bool): True if rare tokens should be replaced with a special unknown
            token. Threshold for considering tokens rare can be found at `saber.constants.NUM_RARE`.
    """
    def __init__(self, dataset_folder=None, replace_rare_tokens=False):
        super(CoNLL2004DatasetReader, self).__init__(dataset_folder, replace_rare_tokens)

    # TODO (John): Newline counter logic will append extra list if end of the file has two newlines.
    def _get_type_seq(self):
        """Loads sequence data from a CoNLL format data set given at `self.dataset_folder`.

        For the CoNLL formatted dataset given at `self.dataset_folder`, updates `self.type_seq` with
        lists containing the word, character and tag sequences for the train and, if provided,
        valid/test partitions found at `self.dataset_folder/train.*`, `self.dataset_folder/valid.*`
        and `self.dataset_folder/test.*`.
        """
        for partition, filepath in self.dataset_folder.items():
            if filepath is not None:
                word_seq = [[]]
                char_seq = [[]]
                ent_seq = [[]]
                rel_seq = [[]]

                newline_counter = 0
                with open(filepath, 'r') as f:
                    for line in f:
                        # If we have seen two newlines, then this is a new sentence
                        if newline_counter == 2:
                            word_seq.append([])
                            char_seq.append([])
                            ent_seq.append([])
                            rel_seq.append([])

                            newline_counter = 0

                        if line not in ['\n', '\r\n']:
                            columns = line.strip().split('\t')
                            # If more than three columns, this is an ent, otherwise it is a rel
                            if len(columns) > 3:
                                _, ent, _, _, _, token, _, _, _ = columns

                                word_seq[-1].append(token)
                                char_seq[-1].append(list(token))
                                ent_seq[-1].append(ent)
                            else:
                                head, tail, rel = columns
                                rel_seq[-1].append([int(head), int(tail), rel])
                        else:
                            newline_counter += 1

                self.type_seq[partition] = {
                    'word': word_seq,
                    'char': char_seq,
                    'ent': ent_seq,
                    'rel': rel_seq,
                }
