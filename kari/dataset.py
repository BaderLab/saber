import os
import time
import glob
import codecs
import numpy as np
import pandas as pd

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# TODO (John): set max_seq_len empirically.
# TODO (John): Do I need to add a pad to char types?

TRAIN_FILE_EXT = 'train.*'
# TEST_FILE_EXT = 'test.*'

# method naming conventions: https://stackoverflow.com/questions/8689964/why-do-some-functions-have-underscores-before-and-after-the-function-name#8689983
class Dataset(object):
    """ A class for handling data sets. """
    def __init__(self, dataset_folder, sep='\t', names=['Word', 'Tag'],
                 header=None, max_seq_len=50):
        self.dataset_folder = dataset_folder
        # search for any files in the dataset filepath ending with
        # TRAIN_FILE_EXT or TEST_FILE_EXT
        self.trainset_filepath = glob.glob(os.path.join(dataset_folder, TRAIN_FILE_EXT))[0]
        # self.testset_filepath = glob.glob(os.path.join(dataset_folder, TEST_FILE_EXT))[0]
        self.sep = sep
        self.names = names
        self.header = header
        self.max_seq_len = max_seq_len
        # shared by train/test
        self.raw_dataframe = None
        self.word_types = []
        self.char_types = []
        self.tag_types = []
        self.word_type_count = 0
        self.char_type_count = 0
        self.tag_type_count = 0
        self.word_type_to_idx = {}
        self.char_type_to_idx = {}
        self.tag_type_to_idx = {}
        # not shared by train/test
        # self.train_dataframe = None
        # self.test_dataframe = None
        self.train_sentences = []
        # self.test_sentences = []
        self.train_word_idx_sequence = []
        # self.test_word_idx_sequence = []
        self.train_tag_idx_sequence = []
        # self.test_tag_idx_sequence = []

        # on object construction, we read the dataset files and get word/tag
        # types, this is helpful for creating compound datasets

        # load dataset file, then grab the train and test 'frames'
        self.raw_dataframe = self._load_dataset()
        # self.train_dataframe = self.raw_dataframe.loc['train']
        # self.test_dataframe = self.raw_dataframe.loc['test']
        # get types and type counts
        self.word_types, self.char_types, self.tag_types = self._get_types()
        self.word_type_count = len(self.word_types)
        self.char_types_count = len(self.char_types)
        self.tag_type_count = len(self.tag_types)

    def load_dataset(self, shared_word_type_to_idx=None, shared_char_type_to_idx=None):
        """ Coordinates loading of given data set at self.dataset_folder.

        For a given dataset in CoNLL format at dataset_folder, cordinates
        the loading of data into a pandas dataframe and updates instance
        attributes. Expects self.dataset_folder to be a directory containing
        a single file, train.*. If this is a compound dataset,
        shared_word_type_to_idx and shared_char_type_to_idx must be provided.

        Args:
            shared_word_type_to_idx: optional, a mapping of word types to
                                     indices shared between datasets
            shared_char_type_to_idx: optional, a mapping of character types to
                                     indices shared between datasets
        """
        # generate type to index mappings
        self.word_type_to_idx, self.tag_type_to_idx = self._map_type_to_idx()

        # if shared_by_compound is passed into function call, then this is a
        # compound dataset (word_type_to_idx is shared across datasets)
        if shared_word_type_to_idx is not None and shared_char_type_to_idx is not None:
            self.word_type_to_idx = shared_word_type_to_idx
            self.char_type_to_idx = shared_char_type_to_idx


        ## TRAIN
        # get sentences
        self.train_sentences = self._get_sentences(self.trainset_filepath, sep=self.sep)
        # get type to idx sequences
        self.train_word_idx_sequence = self._get_type_idx_sequence(self.train_sentences, type_='word')
        self.train_tag_idx_sequence = self._get_type_idx_sequence(self.train_sentences, type_='tag')
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
        """ Loads data set given at self.dataset_folder in pandas dataframe.

        Loads a given dataset in CoNLL format at dataset_folder into a pandas
        dataframe and updates instance. Expects self.dataset_folder to be a
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
                                  # a NA value.
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
        """ Returns word types, char types and tag types.

        Returns the word types, character types and tag types for the current
        loaded dataset (self.raw_dataframe)

        Preconditions:
            the first column of the data set contains the word types, and the
            last column contains the tag types.

        Returns:
            a three-tuple containing the word types, chararcter types and
            tag types.
        """
        # word types
        word_types = list(set(self.raw_dataframe.iloc[:, 0].values))
        word_types.append("ENDPAD")
        # char types
        char_types = []
        for word in word_types:
            char_types.extend(list(word))
        char_types = list(set(char_types))
        # tag types
        tag_types = list(set(self.raw_dataframe.iloc[:, -1].values))

        return word_types, char_types, tag_types

    def _map_type_to_idx(self):
        """ Returns type to index mappings.

        Returns dictionaries mapping each word type and each tag type to a
        unique index.

        Returns:
            two-tuple of word type and tag type to index mappings
        """
        tag_type_to_idx = self._sequence_2_idx(self.tag_types)
        # pad of 1 accounts for the sequence pad (of 0) down the pipeline
        word_type_to_idx = self._sequence_2_idx(self.word_types, pad=1)

        return word_type_to_idx, tag_type_to_idx

    def _sequence_2_idx(self, sequence, offset=0):
        """ Returns a dictionary of element:idx pairs for each element in sequence.

        Given a list, returns a dictionary of length len(sequence) + pad where
        the keys are elements of sequence and the values are unique integers.

        Args:
            sequence: a list of sequence data.
            offset: used when computing the mapping. An offset a 1 means we
                    begin computing the mapping at 1, useful if we want to
                    use 0 as a padding value.
        Returns:
            a mapping from elements in the sequence to numbered indices
        """
        # offset accounts for sequence pad
        return {e: i + pad for i, e in enumerate(list(set(sequence)))}

    @staticmethod
    def sequence_2_idx(sequence, pad=0):
        """ Returns a dictionary of element:idx pairs for each element in sequence.

        A class method that, when given a list, returns a dictionary of length
        len(sequence) + pad where the keys are elements of sequence and the
        values are unique integers.

        Args:
            sequence: a list of sequence data.
            offset: used when computing the mapping. An offset a 1 means we
                    begin computing the mapping at 1, useful if we want to
                    use 0 as a padding value.
        Returns:
            a mapping from elements in the sequence to numbered indices.
        """
        # offset accounts for sequence pad
        return {e: i + pad for i, e in enumerate(list(set(sequence)))}

    def _get_sentences(self, partition_filepath, sep='\t'):
        """ Returns sentences for csv/tsv file as a list lists of tuples.

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

    def _get_type_idx_sequence(self, sentences_, type_):
        """ Returns sequence of indices corresponding to data set sentences.

        Returns the sequence of idices corresponding to the type order in
        sentences_, where type_ can be "word", "char", or "tag" correpsonding to
        word, char and tag types of the Dataset instance.

        Args:
            type_: "word", "char", or "tag"
            sentences: a list of lists, where each list represents a sentence
                for the Dataset instance and each sublist contains ordered
                (word, tag) pairs.

        Returns:
            a list, containing a sequence of idx's corresponding to the type
            order in sentences.

        Preconditions:
            assumes that the first column of the data set contains the word
            types, and the last column contains the tag types.
        """
        assert type_ in ['word', 'char', 'tag'], """parameter type_ must be one
        of 'word', 'char', or 'tag'"""

        # column_idx and pad are different for word types and tag types
        if type_ == 'word':
            column_idx = 0
            # allows us to use the mask_zero parameter of the embedding layer to
            # ignore inputs with value zero.
            pad = 0
            type_to_idx_ = self.word_type_to_idx
        elif type_ == 'char':
            column_idx = 0
            type_to_idx_ = self.char_type_to_idx
        # tag type
        elif type_ == 'tag':
            column_idx = -1
            pad = self.tag_type_to_idx['O']
            type_to_idx_ = self.tag_type_to_idx

        # get sequence of idx's in the order they appear in sentences
        type_sequence = [[type_to_idx_[ty[column_idx]] for ty in s] for s in sentences_]

        # don't pad char sequences
        if type_ != 'char':
            type_sequence = pad_sequences(maxlen=self.max_seq_len,
                                          sequences=type_sequence,
                                          padding='post',
                                          value=pad)

        return type_sequence

    def _tag_idx_sequence_to_categorical(self, idx_sequence):
        """ One-hot encodes a given class vector.

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
        # one_hots = np.squeeze(one_hots, axis=2)
        return one_hots
