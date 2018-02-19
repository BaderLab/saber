import time

import numpy as np
import pandas as pd

from dataset import Dataset
from specify_model import *

# TODO (johngiorgi): set max_seq_len based on empirical observations
# TODO (johngiorgi): consider smarter default values for paramas
# TODO (johngiorgi): make sure this process is shuffling the data

class SequenceProcessingModel(object):
    PARAM_DEFAULT = 'default_value_please_ignore_1YVBF48GBG98BGB8432G4874BF74BB'

    def __init__(self,
                 activation_function=PARAM_DEFAULT,
                 batch_size=PARAM_DEFAULT,
                 dataset_text_folder=PARAM_DEFAULT,
                 debug=PARAM_DEFAULT,
                 dropout_rate=PARAM_DEFAULT,
                 freeze_token_embeddings=PARAM_DEFAULT,
                 gradient_clipping_value=PARAM_DEFAULT,
                 k_folds=PARAM_DEFAULT,
                 learning_rate=PARAM_DEFAULT,
                 maximum_number_of_epochs=PARAM_DEFAULT,
                 model_name=PARAM_DEFAULT,
                 optimizer=PARAM_DEFAULT,
                 output_folder=PARAM_DEFAULT,
                 token_pretrained_embedding_filepath=PARAM_DEFAULT,
                 train_model=PARAM_DEFAULT,
                 max_seq_len=PARAM_DEFAULT
                 ):

        ## INITIALIZE MODEL ATTRIBUTES
        # hyperparameters
        self.activation_function = activation_function
        self.batch_size = batch_size
        self.dataset_text_folder = dataset_text_folder
        self.debug = debug
        self.dropout_rate = dropout_rate
        self.freeze_token_embeddings = freeze_token_embeddings
        self.gradient_clipping_value = gradient_clipping_value
        self.k_folds = k_folds
        self.learning_rate = learning_rate
        self.maximum_number_of_epochs = maximum_number_of_epochs
        self.model_name = model_name
        self.optimizer = optimizer
        self.output_folder = output_folder
        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.train_model = train_model
        self.max_seq_len = max_seq_len
        # dataset tied to the model
        self.ds = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        # embeddings
        self.token_embedding_matrix = None
        # model
        self.model = None
        self.crf = None

        # LOAD DATA
        self.ds = Dataset(self.dataset_text_folder, max_seq_len=self.max_seq_len)
        self.ds.load_dataset()
        # get train/test partitions
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.ds.train_word_idx_sequence,
            self.ds.test_word_idx_sequence,
            self.ds.train_tag_idx_sequence,
            self.ds.test_tag_idx_sequence)
        # if pretrained token embeddings are provided, load them
        if len(token_pretrained_embedding_filepath) > 0:
            self._load_token_embeddings()

        # SPECIFY A MODEL
        if self.model_name == 'LSTM-CRF-NER':
            # pass a dictionary of this the dataset and model objects attributes as
            # argument to specify_
            self.model, self.crf = specify_LSTM_CRF_({**vars(self.ds), **vars(self)})
            compile_LSTM_CRF_(vars(self), self.model, self.crf)

    def fit(self):
        """
        """
        # fit
        train_hist = self.model.fit(self.X_train, self.y_train,
                                    batch_size=self.batch_size,
                                    epochs=self.maximum_number_of_epochs,
                                    validation_split=0.1, verbose=1)

        return pd.DataFrame(train_hist.history)

        '''
        import matplotlib.pyplot as plt
        plt.style.use("ggplot")
        plt.figure(figsize=(12,12))
        plt.plot(hist["acc"])
        plt.plot(hist["val_acc"])
        plt.show()
        '''

    def predict(self):
        """
        """
        # get predicted sequence, flatten into 1D array
        pred_idx = self.model.predict(self.X_test).argmax(axis=-1)
        pred_idx = np.asarray(pred_idx).ravel()
        # get gold sequence, flatten into 1D array
        gold_idx = self.y_test.argmax(axis=-1)
        gold_idx = np.asarray(gold_idx).ravel()

        # get indices for all labels that are not the 'null' label, 'O'.
        labels_ = [(k, v)[1] for k, v in self.ds.tag_type_to_index.items() if k != 'O']

        '''
        print(precision_recall_fscore_support(gold_idx, pred_idx,
                                              # we don't want metrics for 'O' tags
                                              labels=labels_,
                                              average='macro'))
        '''
        print(np.count_nonzero((pred_idx == gold_idx)) / len(pred_idx))

        return pred_idx, gold_idx, labels_

    def _load_token_embeddings(self):
        start_time = time.time()
        print('Loading embeddings... ', end='', flush=True)

        # prepare the embedding indicies
        token_embeddings_index, token_embedding_dimension = self._prepare_token_embedding_layer()
        print('found %s word vectors of dimension %s.' % (len(token_embeddings_index), token_embedding_dimension))
        # fill up the embedding matrix, update attribute
        embedding_matrix = self._prepare_token_embedding_matrix(token_embeddings_index, token_embedding_dimension)
        self.token_embedding_matrix = embedding_matrix

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds)'.format(elapsed_time))

        return

    def _prepare_token_embedding_layer(self):
        """ Creates an embedding index using pretrained token embeddings.

        For the models given pretrained token embeddings, creates and returns a
        dictionary mapping words to known embeddings.

        Returns:
            token_embeddings_index: mapping of words to pre-trained token embeddings
            token_embedding_dimensions: the dimension of the pre-trained token embeddings
        """
        token_embeddings_index = {}
        token_embedding_file_lines = []
        token_embedding_dimensions = 0

        with open(self.token_pretrained_embedding_filepath, 'r') as pte:
            token_embedding_file_lines = pte.readlines()

        for emb in token_embedding_file_lines:
            values = emb.split()

            word = values[0] # get words
            coefs = np.asarray(values[1:], dtype='float32') # get emb vectos

            # get size of token embedding dimensions
            if token_embedding_dimensions == 0:
                token_embedding_dimensions = len(coefs)
            # update our embedding index
            token_embeddings_index[word] = coefs

        return token_embeddings_index, token_embedding_dimensions

    def _prepare_token_embedding_matrix(self,
                                        token_embeddings_index,
                                        token_embedding_dimensions):
        """ Creates an embedding matrix using pretrained token embeddings.

        For the models word to idx mappings, and word to pre-trained token
        embeddings, creates a matrix which maps all words in the models dataset
        to a pre-trained token embedding. If the token embedding does not exist
        in the pre-trained token embeddings file, the word will be mapped to
        an embedding of all zeros.

        Returns:
            token_embeddings_matrix: a matrix whos ith row corresponds to the
            token embedding for the ith word in the models word to idx mapping.
        """
        # initialize the embeddings matrix
        token_embeddings_matrix = np.zeros((len(self.ds.word_type_to_index) + 1,
                                            token_embedding_dimensions))

        # lookup embeddings for every word in the dataset
        for word, i in self.ds.word_type_to_index.items():
            token_embeddings_vector = token_embeddings_index.get(word)
            if token_embeddings_vector is not None:
                # words not found in embedding index will be all-zeros.
                token_embeddings_matrix[i] = token_embeddings_vector

        return token_embeddings_matrix
