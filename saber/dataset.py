import os
import time
import glob
import codecs
import numpy as np

from keras.utils import to_categorical

from constants import UNK
from constants import PAD
from constants import TRAIN_FILE_EXT
from preprocessor import Preprocessor

class Dataset(object):
    """A class for handling data sets."""
    def __init__(self, filepath, sep='\t', names=['Word', 'Tag'], header=None):
        self.filepath = filepath
        # search for any files in the dataset filepath ending with
        # TRAIN_FILE_EXT or TEST_FILE_EXT
        self.trainset_filepath = glob.glob(os.path.join(filepath, TRAIN_FILE_EXT))[0]
        # self.testset_filepath = glob.glob(os.path.join(filepath, TEST_FILE_EXT))[0]
        self.sep = sep
        self.names = names
        self.header = header

        # word and tag sequences from dataset
        self.word_seq = None
        self.tag_seq = None

        # word, character, and tag types from dataset
        self.word_types = None
        self.char_types = None
        self.tag_types = None

        # mappings of type: index for word, character and tag types
        self.word_type_to_idx = None
        self.char_type_to_idx = None
        self.tag_type_to_idx = None
        # inverse mapping from indices to tag types 
        self.idx_to_tag_type = None

        # index sequences of words, characters and tags for training
        self.train_word_idx_seq = None
        self.train_char_idx_seq = None
        self.train_tag_idx_seq = None

        # load dataset file, then grab the train and test 'frames'
        # self.train_dataframe = self.raw_dataframe.loc['train']
        # self.test_dataframe = self.raw_dataframe.loc['test']

    def load_dataset(self, word_type_to_idx=None, char_type_to_idx=None):
        """Coordinates loading of given data set at self.filepath.

        For a given dataset in CoNLL format at filepath, cordinates
        the loading of data into a pandas dataframe and updates instance
        attributes. Expects self.filepathto be a directory containing
        a single file, train.*. If this is a compound dataset,
        word_type_to_idx and char_type_to_idx must be provided.

        Args:
            word_type_to_idx: optional, a mapping of word types to
                                     indices shared between datasets
            char_type_to_idx: optional, a mapping of character types to
                                     indices shared between datasets
        """
        # if shared_by_compound is passed into function call, then this is a
        # compound dataset (word and char index mappings shared across datasets)
        if word_type_to_idx is not None and char_type_to_idx is not None:
            self.word_type_to_idx = word_type_to_idx
            self.char_type_to_idx = char_type_to_idx
        else:
            # load data and labels from file
            self.load_data_and_labels()
            # get word, char, and tag types
            self.get_types()
            # generate shared type to index mappings
            self.word_type_to_idx = Preprocessor.sequence_to_idx(self.word_types)
            self.char_type_to_idx  = Preprocessor.sequence_to_idx(self.char_types)

        # generate un-shared type to index mappings
        self.tag_type_to_idx = Preprocessor.sequence_to_idx(self.tag_types)
        # create reverse mapping of indices to tags
        self.idx_to_tag_type = {v: k for k, v in self.tag_type_to_idx.items()}

        # get type to idx sequences
        self.train_word_idx_seq = Preprocessor.get_type_idx_sequence(self.word_seq, self.word_type_to_idx, type='word')
        self.train_char_idx_seq = Preprocessor.get_type_idx_sequence(self.word_seq, self.char_type_to_idx, type='char')
        self.train_tag_idx_seq = Preprocessor.get_type_idx_sequence(self.tag_seq, self.tag_type_to_idx, type='tag')
        # convert tag idx sequences to categorical
        self.train_tag_idx_seq = self._tag_idx_sequence_to_categorical(self.train_tag_idx_seq)

    def load_data_and_labels(self):
        """Loads CoNLL format data set given at self.trainset_filepath.

        Updates self.word_seq and self.tag_seq each with a numpy array of lists
        containg the word and tag sequence respectively. Expects filepath to be
        a directory containing one file, train.*. The file format is
        tab-separated values. A blank line is required at the end of each
        sentence.

        Example dataset:
        '''
        The	O
        transcription	O
        of	O
        most	O
        RP	B-PRGE
        genes	I-PRGE
        is	O
        activated	O
        by	O
        ...
        '''
        """
        # global accumulators
        word_seq, tag_seq = [], []

        with codecs.open(self.trainset_filepath, 'r', encoding='utf-8') as ds:
            # local accumulators
            words, tags = [], []

            for line in ds:
                line = line.rstrip() # right strip

                # accumulate each sequence
                if len(line) == 0 or line.startswith('-DOCSTART-'):
                    if len(words) != 0:
                        word_seq.append(words)
                        tag_seq.append(tags)
                        words, tags = [], []

                # accumulate each word in a sequence
                else:
                    word, tag = line.split('\t')
                    words.append(word)
                    tags.append(tag)

            # in order to collect last sentence in the file
            word_seq.append(words)
            tag_seq.append(tags)

        self.word_seq = np.asarray(word_seq)
        self.tag_seq = np.asarray(tag_seq)

    def get_types(self):
        """Returns word types, char types and tag types.

        Returns the word types, character types and tag types for the current
        loaded dataset (self.raw_dataframe)

        Preconditions:
            the first column of the data set contains the word types, and the
            last column contains the tag types.

        Returns:
            a three-tuple containing the word types, chararcter types and
            tag types.
        """
        # Word types
        word_types = list(set([w for s in self.word_seq for w in s]))

        # Char types
        char_types = []
        for word in word_types:
            char_types.extend(list(word))
        char_types = list(set(char_types))

        # Tag types
        tag_types = list(set([t for s in self.tag_seq for t in s]))

        # Post processing
        word_types.insert(0, PAD) # make PAD the 0th index
        word_types.insert(1, UNK) # make UNK the 1st idex
        char_types.insert(0, PAD)
        char_types.insert(1, UNK)
        tag_types.insert(0, PAD)

        self.word_types = word_types
        self.char_types = char_types
        self.tag_types = tag_types

    def _map_type_to_idx(self):
        """Returns type to index mappings.

        Returns dictionaries mapping each word, char and tag type to a unique
        index.

        Returns:
            three-tuple of word, char and tag type to index mappings
        """
        self.word_type_to_idx = Preprocessor.sequence_to_idx(self.word_types)
        self.char_type_to_idx  = Preprocessor.sequence_to_idx(self.char_types)
        self.tag_type_to_idx = Preprocessor.sequence_to_idx(self.tag_types)

    def _tag_idx_sequence_to_categorical(self, idx_seq):
        """One-hot encodes a given class vector.

        Converts a class vector of integers, (idx_sequence) to a
        one-hot encoded matrix. The number of classes in the one-hot encoding
        determined by self.tag_type_count.

        Args:
            idx_sequence: a class vector of integers, representing a sequence
                          of tags.

        Returns:
            one-hot endcoded matrix representation of idx_sequence.
        """
        # convert to one-hot encoding
        one_hots = [to_categorical(s, len(self.tag_types)) for s in idx_seq]
        one_hots = np.array(one_hots)

        return one_hots
