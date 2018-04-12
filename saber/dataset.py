import os
import time
import glob
import codecs
import numpy as np
import pandas as pd

from keras.utils import to_categorical

from preprocessor import Preprocessor

TRAIN_FILE_EXT = 'train.*'
# TEST_FILE_EXT = 'test.*'
ENDPAD = 'ENDPAD'

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

        # shared by train/test
        self.word_type_to_idx = {}
        self.char_type_to_idx = {}
        self.tag_type_to_idx = {}
        # on object construction, we read the dataset files and get word/tag
        # types, this is helpful for creating compound datasets
        self.raw_dataframe = self._load_dataset()
        self.word_types, self.char_types, self.tag_types = self._get_types()
        self.word_type_count = len(self.word_types)
        self.char_type_count = len(self.char_types)
        self.tag_type_count = len(self.tag_types)

        # not shared by train/test
        self.train_sentences = []
        self.train_word_idx_sequence = []
        self.train_char_idx_sequence = []
        self.train_tag_idx_sequence = []

        # load dataset file, then grab the train and test 'frames'
        # self.train_dataframe = self.raw_dataframe.loc['train']
        # self.test_dataframe = self.raw_dataframe.loc['test']

    def load_dataset(self, shared_word_type_to_idx=None, shared_char_type_to_idx=None):
        """ Coordinates loading of given data set at self.filepath.

        For a given dataset in CoNLL format at filepath, cordinates
        the loading of data into a pandas dataframe and updates instance
        attributes. Expects self.filepathto be a directory containing
        a single file, train.*. If this is a compound dataset,
        shared_word_type_to_idx and shared_char_type_to_idx must be provided.

        Args:
            shared_word_type_to_idx: optional, a mapping of word types to
                                     indices shared between datasets
            shared_char_type_to_idx: optional, a mapping of character types to
                                     indices shared between datasets
        """
        # generate type to index mappings
        self.word_type_to_idx, self.char_type_to_idx, self.tag_type_to_idx = \
            self._map_type_to_idx()

        # if shared_by_compound is passed into function call, then this is a
        # compound dataset (word_type_to_idx is shared across datasets)
        if ((shared_word_type_to_idx is not None) and
           (shared_char_type_to_idx is not None)):
            self.word_type_to_idx = shared_word_type_to_idx
            self.char_type_to_idx = shared_char_type_to_idx

        ## TRAIN
        # get sentences
        self.train_sentences = self._get_sentences(self.trainset_filepath, sep=self.sep)
        # get type to idx sequences
        self.train_word_idx_sequence = Preprocessor.get_type_idx_sequence(self.train_sentences,
                                                                          word_type_to_idx=self.word_type_to_idx)
        self.train_char_idx_sequence = Preprocessor.get_type_idx_sequence(self.train_sentences,
                                                                          char_type_to_idx=self.char_type_to_idx)
        self.train_tag_idx_sequence = Preprocessor.get_type_idx_sequence(self.train_sentences,
                                                                          tag_type_to_idx=self.tag_type_to_idx)
        # convert tag idx sequences to categorical
        self.train_tag_idx_sequence = self._tag_idx_sequence_to_categorical(self.train_tag_idx_sequence)
        ## TEST
        # get sentences
        # self.test_sentences = self._get_sentences(self.testset_filepath, sep=self.sep)
        # get type to idx sequences
        # self.test_word_idx_sequence = self._get_type_idx_sequence(self.test_sentences)
        # self.test_tag_idx_sequence = self._get_type_idx_sequence(self.test_sentences, type_='tag')
        # convert tag idx sequences to categorical
        # self.test_tag_idx_sequence = self._tag_idx_sequence_to_categorical(self.test_tag_idx_sequence)

    def _load_dataset(self):
        """ Loads data set given at self.filepathin pandas dataframe.

        Loads a given dataset in CoNLL format at filepathinto a pandas
        dataframe and updates instance. Expects self.filepathto be a
        directory containing two files, train.* and test.*

        Returns:
            single merged pandas dataframe for train and test files.
        """
        raw_dataset = pd.read_csv(self.trainset_filepath,
                                  header=self.header, sep=self.sep,
                                  names=self.names, encoding="utf-8",
                                  # forces pandas to ignore quotes such that we
                                  # can read in '"' word type.
                                  quoting = 3,
                                  # prevents pandas from interpreting 'null' as
                                  # a NA value, as in e.g. 'null hypotheis'
                                  na_filter=False)

        # testing_set = pd.read_csv(self.testset_filepath,
        #                           header=self.header, sep=self.sep,
        #                           names=self.names, encoding="utf-8",
        #                           quoting = 3, na_filter=False)

        # concatenate dataframes vertically with keys
        # frames = [training_set, testing_set]
        # raw_dataset = pd.concat(frames, keys=['train', 'test'])
        # forward propogate last valid value to file NAs.
        raw_dataset = raw_dataset.fillna(method='ffill')
        return raw_dataset

    def _get_types(self):
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
        word_types = list(set(self.raw_dataframe.iloc[:, 0].values))
        word_types.insert(0, ENDPAD) # make ENDPAD the 0th element

        # Char types
        char_types = []
        for word in word_types:
            char_types.extend(list(word))
        char_types = list(set(char_types))

        # Tag types
        tag_types = list(set(self.raw_dataframe.iloc[:, -1].values))
        # if the negative class is not first, swap it with the first element
        neg_class_idx = tag_types.index('O')
        tag_types[neg_class_idx], tag_types[0] = tag_types[0], tag_types[neg_class_idx]

        return word_types, char_types, tag_types

    def _map_type_to_idx(self):
        """ Returns type to index mappings.

        Returns dictionaries mapping each word type and each tag type to a
        unique index.

        Returns:
            three-tuple of word, char and tag type to index mappings
        """
        # TODO (johngiorgi): Why did I drop the offset?
        word_type_to_idx = Preprocessor.sequence_to_idx(self.word_types)
        char_type_to_idx = Preprocessor.sequence_to_idx(self.char_types)
        tag_type_to_idx = Preprocessor.sequence_to_idx(self.tag_types)

        return word_type_to_idx, char_type_to_idx, tag_type_to_idx

    def _get_sentences(self, partition_filepath, sep='\t'):
        """Returns sentences for csv/tsv file as a list lists of tuples.

        Returns a list of lists of two-tuples for the csv/tsv at
        partition_filepath, where the inner lists represent sentences, and the
        tuples are ordered (word, tag) pairs.

        Args:
            partition_filepath: filepath to csv/tsv file for train/test set partition
            sep: delimiter for csv/tsv file at filepath

        Returns:
            a list of lists of two-tuples, where the inner lists are sentences
            containing ordered (word, tag) pairs.
        """
        # accumulators
        global_sentence_acc = []
        indivdual_sentence_acc = []

        with codecs.open(partition_filepath, 'r', encoding='utf-8') as ds:
            for line in ds:
                # accumulate all lines in a sentence
                if line != '\n':
                    indivdual_sentence_acc.append(tuple(line.strip().split('\t')))
                # reached the end of the sentence
                else:
                    global_sentence_acc.append(indivdual_sentence_acc)
                    indivdual_sentence_acc = []

            # in order to collect last sentence in the file
            global_sentence_acc.append(indivdual_sentence_acc)

        return global_sentence_acc

    def _tag_idx_sequence_to_categorical(self, idx_sequence):
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
        one_hots = [to_categorical(seq, self.tag_type_count) for seq in idx_sequence]
        one_hots = np.array(one_hots)

        return one_hots
