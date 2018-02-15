import time
import codecs
import csv
import pandas as pd

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# TODO (John): _get_types is dropping quotation marks when getting word types,
# why??
# TODO (John): set max_seq_len empirically.

# method naming conventions: https://stackoverflow.com/questions/8689964/why-do-some-functions-have-underscores-before-and-after-the-function-name#8689983
class Dataset(object):
    """ A class for handling data sets. """
    def __init__(self, dataset_filepath, sep='\t', names=['Word', 'Tag'],
                 header=None, max_seq_len=75):
        self.dataset_filepath = dataset_filepath
        self.sep = sep
        self.names = names
        self.header = header
        self.max_seq_len = max_seq_len

        self.raw_dataframe = None
        self.word_types = []
        self.tag_types = []
        self.word_type_count = 0
        self.tag_type_count = 0
        self.word_type_to_index = {}
        self.tag_type_to_index = {}
        self.sentences = []
        self.word_idx_sequence = []
        self.tag_idx_sequence = []

    def load_dataset(self):
        """ Loads data set in CoNLL <?> format.

        For a given dataset in CoNLL <?> format at dataset_filepath, loads data
        into a pandas dataframe and updates instance attributes.
        """
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)

        self.raw_dataframe = self._load_dataset()
        self.word_types, self.tag_types = self._get_types()
        self.word_type_count, self.tag_type_count = len(self.word_types), len(self.tag_types)
        self.word_type_to_index, self.tag_type_to_index = self._map_type_to_idx()
        self.sentences = self._get_sentences(sep=self.sep)
        self.word_idx_sequence = self._get_type_idx_sequence(self.word_type_to_index,
                                                             self.sentences)
        self.tag_idx_sequence = self._get_type_idx_sequence(self.tag_type_to_index,
                                                            self.sentences,
                                                            type_='tag')
        self.tag_idx_sequence = [to_categorical(i, num_classes=self.tag_type_count)
                                 for i in self.tag_idx_sequence]

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

    def _load_dataset(self):
        dataset = pd.read_csv(self.dataset_filepath, header=self.header,
                              sep=self.sep, names=self.names, encoding="utf-8",
                              # forces pandas to ignore quotes such that we
                              # can read in '"' word type.
                              quoting = 3,
                              # prevents pandas from interpreting 'null' as a
                              # NA value.
                              na_filter=False)

        # forward propogate last valid value to file NAs.
        dataset = dataset.fillna(method='ffill')
        return dataset

    def _get_types(self):
        """ Updates attributes with word types, class types and their counts.

        Updates the attributes of a Dataset object instance with the datasets
        word types, tag types, and counts for these entities.

        Preconditions:
            assumes that the first column of the data set contains the word
            types, and the last column contains the tag types.
        """
        word_types = list(set(self.raw_dataframe.iloc[:, 0].values))
        tag_types = list(set(self.raw_dataframe.iloc[:, -1].values))
        word_types.append("ENDPAD")

        return word_types, tag_types

    def _map_type_to_idx(self):
        """ Updates attributes with type to index dictionary maps.

        Updates the attributes of a Dataset object instance with dictionaries
        mapping each word type and each tag type to a unique index.
        """
        tag_type_to_index = self._sequence_2_idx(self.tag_types)
        # pad of 1 accounts for the sequence pad (of 0) down the pipeline
        word_type_to_index = self._sequence_2_idx(self.word_types, pad=1)

        return word_type_to_index, tag_type_to_index

    def _sequence_2_idx(self, sequence, pad=0):
        """ Returns a dictionary of element:idx pairs for each element in sequence.

        Given a list, returns a dictionary of length len(sequence) + pad where
        the keys are elements of sequence and the values are unique integers.

        Args:
            sequence: a list of sequence data.
        """
        # pad accounts for idx of sequence pad
        return {e: i + pad for i, e in enumerate(sequence)}

    def _get_sentences(self, sep='\t'):
        """
        """
        master_sentence_acc = []
        indivdual_sentence_acc = []

        with codecs.open(self.dataset_filepath, 'r', encoding='utf-8') as ds:
            for line in ds:
                if line != '\n':
                    indivdual_sentence_acc.append(tuple(line.strip().split('\t')))
                else:
                    master_sentence_acc.append(indivdual_sentence_acc)
                    indivdual_sentence_acc = []

            # in order to collect last sentence in the file
            master_sentence_acc.append(indivdual_sentence_acc)

        return master_sentence_acc

    def _get_type_idx_sequence(self, type_to_idx, sentences, type_='word'):
        """ Returns sequence of idicies corresponding to data set sentences.

        Given a dictionary of type:idx key, value pairs, returns the sequence
        of idx corresponding to the type order in sentences. Type can be the
        word types or tag types of Dataset instance.

        Args:
            types: a dictionary of type, idx key value pairs
            sentences: a list of lists, where each list represents a sentence for
                the Dataset instance and each sublist contains tuples of type
                and tag.

        Returns:
            a list, containing a sequence of idx's corresponding to the type
            order in sentences.

        Preconditions:
            assumes that the first column of the data set contains the word
            types, and the last column contains the tag types.
        """
        # column_idx and pad are different for word types and tag types
        # word type
        column_idx = 0
        pad = 0
        # tag type
        if type_ == 'tag':
            column_idx = -1
            pad = self.tag_type_to_index['O']

        # get sequence of idx's in the order they appear in sentences
        type_sequence = [[type_to_idx[ty[column_idx]] for ty in s] for s in self.sentences]
        type_sequence = pad_sequences(maxlen=self.max_seq_len, sequences=type_sequence,
                                      padding='post', value=pad)
        return type_sequence
