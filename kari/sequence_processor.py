import os
import time

import numpy as np
import pandas as pd

from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from dataset import Dataset
from metrics import Metrics
from utils_generic import make_dir

print('Kari version: {0}'.format('0.1-dev'))

# TODO (johngiorgi): make model checkpointing a config param
# TODO (johngiorgi): make a debug mode that doesn't load token embeddings and
# loads only some lines of dataset
# TODO (johngiorgi): implement saving loading of models
# TODO (johngiorgi): predict should be more of an interface, calling it should
# return a nicely formatted representation of the prediticted entities.

class SequenceProcessor(object):
    """A class for handeling the loading, saving, training, and specifying of
    sequence processing models. """

    def __init__(self, config):
        # hyperparameters
        self.config = config

        # dataset(s) tied to this instance
        self.ds = []
        # token embeddings tied to this instance
        self.token_embedding_matrix = None

        # Keras model object tied to this instance
        self.model = None

    def load_pretrained_model_(self, pretrained_model_filepath):
        """
        """
        # use with keras_contrib.utils
        # save_load_utils.save_all_weights(self.model, path)
        # save_load_utils.load_all_weights(self.model, path)

        self.model = load_model(pretrained_model_filepath)

    def load_dataset(self):
        """Coordinates the loading of a dataset.

        Coordinates the loading of a dataset by creating a one or more Dataset
        objects (one for each filepath in self.dataset_folder). Additionaly,
        if self.token_pretrained_embedding_filepath is provided, loads the token
        embeddings.
        """
        assert len(self.config['dataset_folder']) > 0, '''You must provide at
        least one dataset via the dataset_folder parameter'''

        start_time = time.time()
        # Datasets may be 'single' or 'compound' (more than one), loading
        # differs slightly. Consider a dataset single if there is only one
        # filepath in self.config['dataset_folder'] and compound otherwise.
        if len(self.config['dataset_folder']) == 1:
            print('[INFO] Loading (single) dataset... ', end='', flush=True)
            self.ds = self._load_single_dataset()
        else:
            print('[INFO] Loading (compound) dataset... ', end='', flush=True)
            self.ds = self._load_compound_dataset()

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds)'.format(elapsed_time))

        # if pretrained token embeddings are provided, load them (if they are
        # not already loaded)
        if (len(self.config['token_pretrained_embedding_filepath']) > 0 and
                self.token_embedding_matrix is None):
            self._load_token_embeddings()

    def create_model(self):
        """Specifies and compiles chosen model (self.config['model_name'])."""
        # setup the chosen model
        if self.config['model_name'] == 'LSTM-CRF':
            print('Building the single-task LSTM-CRF model for NER...', end='', flush=True)
            from models.simple_lstm_crf_ner import SimpleLSTMCRF
            model_ = SimpleLSTMCRF(config=self.config,
                                   ds=self.ds,
                                   token_embedding_matrix=self.token_embedding_matrix)
        elif self.config['model_name'] == 'MT-LSTM-CRF':
            print('[INFO] Building the multi-task LSTM-CRF model...')
            from models.multi_task_lstm_crf import MultiTaskLSTMCRF
            model_ = MultiTaskLSTMCRF(config=self.config,
                                      ds=self.ds,
                                      token_embedding_matrix=self.token_embedding_matrix)

        # specify and compile the chosen model
        model_.specify_()
        model_.compile_()
        # update this objects model attribute with instance of model class
        self.model = model_

        print('Done', flush=True)

    def fit(self):
        """Fit the specified model.

        For the given model (self.model), sets up per epoch checkpointing
        and fits the model.

        Returns:
            train_hist, the history of the model training as a pandas
            dataframe.
        """
        # create a Callback object for model checkpointing
        checkpointer = self._setup_model_checkpointing()
        # fit
        # train_history = self.model.fit_(checkpointer=checkpointer)
        # don't get history for now
        self.model.fit_(checkpointer=checkpointer)
        '''
        # create Callback object for per epoch prec/recall/f1/support metrics
        metrics = Metrics(self.X_train, self.y_train, self.ds.word_type_to_idx)
        # fit
        train_history = self.model.fit(self.X_train,
                                       self.y_train,
                                       batch_size=self.batch_size,
                                       epochs=self.maximum_number_of_epochs,
                                       validation_split=0.1,
                                       callbacks = [checkpointer, metrics],
                                       verbose=1)
        '''
        # train_history = pd.DataFrame(train_history.history)
        # return train_history

    def predict(self, task=0):
        """Performs prediction for a given model and returns results.

        Performs prediction for the current model (self.model), and returns
        a 2-tuple contain 1D array-like objects containing the true (gold)
        labels and the predicted labels, where labels are integers corresponding
        to the sequence tags as per self.ds.word_type_to_idx.

        Returns:
            y_true: 1D array like object containing the gold label sequence.
            y_pred: 1D array like object containing the predicted sequence.
        """
        X = self.ds[task].train_word_idx_sequence
        y = self.ds[task].train_tag_idx_sequence
        # get gold sequence, flatten into 1D array
        y_true = y.argmax(axis=-1)
        # y_true = np.asarray(y_true).ravel()
        # get predicted sequence, flatten into 1D array
        y_pred = self.model[task].model.predict(X).argmax(axis=-1)
        # y_pred = np.asarray(y_pred).ravel()

        return y_true, y_pred

    def _load_single_dataset(self):
        """Loads a single dataset.

        Creates and loads a single dataset object for a dataset at
        self.dataset_folder[0].

        Returns:
            a list containing a single dataset object.
        """
        ds = Dataset(dataset_folder=self.config['dataset_folder'][0],
                     max_seq_len=self.config['max_seq_len'])
        ds.load_dataset()

        return [ds]

    def _load_compound_dataset(self):
        """Loads a compound dataset.

        Creates and loads a 'compound' dataset. Compound datasets are specified
        by multiple individual datasets, and share multiple attributes
        (such as word/char type to index mappings). Loads such a dataset for
        each dataset at self.dataset_folder[0].

        Returns:
            A list containing multiple compound dataset objects.
        """
        # accumulator for datasets
        ds_acc = []

        for ds_filepath in self.config['dataset_folder']:
            ds_acc.append(Dataset(ds_filepath, max_seq_len=self.config['max_seq_len']))

        # get combined set of word types from all datasets
        comb_word_types = []
        comb_char_types = []
        for ds in ds_acc:
            comb_word_types.extend(ds.word_types)
            comb_char_types.extend(ds.char_types)
        comb_word_types = list(set(comb_word_types))
        comb_char_types = list(set(comb_char_types))

        # compute word to index mappings that will be shared across datasets
        # pad of 1 accounts for the sequence pad (of 0) down the pipeline
        shared_word_type_to_idx = Dataset.sequence_2_idx(comb_word_types)
        shared_char_type_to_idx = Dataset.sequence_2_idx(comb_char_types)

        # load all the datasets
        for ds in ds_acc:
            ds.load_dataset(shared_word_type_to_idx=shared_word_type_to_idx,
                            shared_char_type_to_idx=shared_char_type_to_idx)

        return ds_acc

    def _setup_model_checkpointing(self):
        """Sets up per epoch model checkpointing.

        Sets up model checkpointing by:
            1) creating the output_folder if it does not already exists.
            2) creating the checkpointing CallBack Keras object.

        Returns:
            checkpointer: a Keras CallBack object for per epoch model
                checkpointing.
        """
        # create output directory if it does not exist
        make_dir(self.config['output_folder'])
        # create path to output folder
        output_folder_ = os.path.join(self.output_folder,
                                      'model_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')
        # set up model checkpointing
        checkpointer = ModelCheckpoint(filepath=output_folder_)

        return checkpointer

    def _load_token_embeddings(self):
        """Coordinates the loading of pre-trained token embeddings.

        Coordinates the loading of pre-trained token embeddings by reading in
        the file containing the token embeddings and created an embedding matrix
        whos ith row corresponds to the token embedding for the ith word in the
        models word to idx mapping.
        """
        start_time = time.time()
        print('[INFO] Loading embeddings... ', end='', flush=True)

        # prepare the embedding indicies
        token_embeddings_index = self._prepare_token_embedding_layer()
        token_embedding_dimension = len(list(token_embeddings_index.values())[0])
        # fill up the embedding matrix, update attribute
        embedding_matrix = self._prepare_token_embedding_matrix(token_embeddings_index, token_embedding_dimension)
        self.token_embedding_matrix = embedding_matrix

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds)'.format(elapsed_time))
        print('Found {} word vectors of dimension {}'.format(len(token_embeddings_index), token_embedding_dimension))

    def _prepare_token_embedding_layer(self):
        """ Creates an embedding index using pretrained token embeddings.

        For the models given pretrained token embeddings, creates and returns a
        dictionary mapping words to known embeddings.

        Returns:
            token_embeddings_index: mapping of words to pre-trained token embeddings
        """
        token_embeddings_index = {}
        token_embedding_file_lines = []

        with open(self.config['token_pretrained_embedding_filepath'], 'r') as pte:
            token_embedding_file_lines = pte.readlines()

        for emb in token_embedding_file_lines:
            values = emb.split()

            word = values[0] # get words
            coefs = np.asarray(values[1:], dtype='float32') # get emb vectos

            # update our embedding index
            token_embeddings_index[word] = coefs

        return token_embeddings_index

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
            token_embedding_matrix: a matrix whos ith row corresponds to the
            token embedding for the ith word in the models word to idx mapping.
        """
        # initialize the embeddings matrix
        token_embedding_matrix = np.zeros((len(self.ds[0].word_type_to_idx) + 1,
                                           token_embedding_dimensions))

        # lookup embeddings for every word in the dataset
        for word, i in self.ds[0].word_type_to_idx.items():
            token_embeddings_vector = token_embeddings_index.get(word)
            if token_embeddings_vector is not None:
                # words not found in embedding index will be all-zeros.
                token_embedding_matrix[i] = token_embeddings_vector

        return token_embedding_matrix
