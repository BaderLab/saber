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

# TODO (johngiorgi): set max_seq_len based on empirical observations
# TODO (johngiorgi): consider smarter default values for paramas
# TODO (johngiorgi): make sure this process is shuffling the data
# TODO (johngiorgi): make model checkpointing a config param
# TODO (johngiorgi): make a debug mode that doesnt load token embeddings and loads only some lines of dataset
# TODO (johngiorgi): abstract away all dataset details as single object

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
                 load_pretrained_model=PARAM_DEFAULT,
                 maximum_number_of_epochs=PARAM_DEFAULT,
                 model_name=PARAM_DEFAULT,
                 optimizer=PARAM_DEFAULT,
                 output_folder=PARAM_DEFAULT,
                 pretrained_model_weights=PARAM_DEFAULT,
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
        self.load_pretrained_model = load_pretrained_model
        self.maximum_number_of_epochs = maximum_number_of_epochs
        self.model_name = model_name
        self.optimizer = optimizer
        self.output_folder = output_folder
        self.pretrained_model_weights = pretrained_model_weights
        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.train_model = train_model
        self.max_seq_len = max_seq_len
        # dataset(s) tied to this instance
        self.ds = []
        # embeddings tied to this instance
        self.token_embedding_matrix = None
        # Keras model object tied to this instance
        self.model = None

    def load_pretrained_model_(self, pretrained_model_filepath):
        """
        """
        self.model = load_model(pretrained_model_filepath)

    def load_dataset(self):
        """ Coordinates the loading of a dataset.

        Coordinates the loading of a dataset by creating a one or more Dataset
        objects (one for each filepath in self.dataset_filepath). Additionaly,
        if self.token_pretrained_embedding_filepath is provided, loads the token
        embeddings.
        """
        assert len(self.dataset_text_folder) > 0, '''You must provide at
        least one dataset via the dataset_text_folder parameter'''

        start_time = time.time()
        # Datasets may be 'single' or 'compound' (more than one), loading
        # differs slightly. Consider a dataset single if there is only one
        # filepath in self.dataset_text_folder and compound otherwise.
        if len(self.dataset_text_folder) == 1:
            print('Loading (single) dataset... ', end='', flush=True)
            self.ds = self._load_single_dataset()
        else:
            print('Loading (compound) dataset... ', end='', flush=True)
            self.ds = self._load_compound_dataset()

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds)'.format(elapsed_time))

        # if pretrained token embeddings are provided, load them (if they are
        # not already loaded)
        if (len(self.token_pretrained_embedding_filepath) > 0 and
            self.token_embedding_matrix is None):
            self._load_token_embeddings()

    def specify_model(self):
        """ Specifies and compiles the chosen model (self.model_name). """
        # create a dictionary of the SequenceProcessingModel objects attributes
        model_specifications = vars(self)

        # setup the chosen model
        if self.model_name == 'LSTM-CRF-NER':
            print('Building the simple LSTM-CRF model for NER...', end='', flush=True)
            from models.simple_lstm_crf_ner import SimpleLSTMCRF
            model_ = SimpleLSTMCRF(model_specifications=model_specifications)
        elif self.model_name == 'MT-LSTM-CRF':
            print('Building the multi-task LSTM-CRF model...', end='', flush=True)
            from models.multi_task_lstm_crf import MultiTaskLSTMCRF
            model_ = MultiTaskLSTMCRF(model_specifications=model_specifications)

        # specify and compile the chosen model
        model_.specify_()
        model_.compile_()
        # update this objects model attribute with the compiled Keras model
        self.model = model_

        print('Done', flush=True)

    def fit(self):
        """ Fit the specified model.

        For the given model (self.model), sets up per epoch checkpointing
        and fits the model.

        Returns:
            train_hist, the history of the model training as a pandas
            dataframe.
        """
        train_history = self.model.fit_(self.ds)
        '''
        # create a Callback object for model checkpointing
        checkpointer = self._setup_model_checkpointing()
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
        train_history = pd.DataFrame(train_history.history)

        return train_history

    def predict(self):
        """ Performs prediction for a given model and returns results.

        Performs prediction for the current model (self.model), and returns
        a 2-tuple contain 1D array-like objects containing the true (gold)
        labels and the predicted labels, where labels are integers corresponding
        to the sequence tags as per self.ds.word_type_to_idx.

        Returns:
            y_true: 1D array like object containing the gold label sequence
            y_pred: 1D array like object containing the predicted sequence
        """
        # get gold sequence, flatten into 1D array
        y_true = self.y_test.argmax(axis=-1)
        y_true = np.asarray(gold_idx).ravel()
        # get predicted sequence, flatten into 1D array
        y_pred = self.model.predict(self.X_test).argmax(axis=-1)
        y_pred = np.asarray(pred_idx).ravel()

        # get indices for all labels that are not the 'null' label, 'O'.
        # labels_ = [(k, v)[1] for k, v in self.ds.word_type_to_idx.items() if k != 'O']

        return y_true, y_pred

    def _load_single_dataset(self):
        """ Loads a single dataset.

        Creates and loads a single dataset object for a dataset at
        self.dataset_text_folder[0].

        Returns:
            a list containing a single dataset object
        """
        ds = Dataset(self.dataset_text_folder[0], max_seq_len=self.max_seq_len)
        ds.load_dataset()

        return [ds]

    def _load_compound_dataset(self):
        """ Loads a compound dataset.

        Creates and loads multiple, 'compound' datasets. Compound datasets
        share multiple attributes (such as word/tag type to index mappings).
        Loads such a dataset for each dataset at self.dataset_text_folder[0].

        Returns:
            a list containing multipl compound dataset objects
        """
        # accumulator for datasets
        ds_acc = []

        for ds_filepath in self.dataset_text_folder:
            ds_acc.append(Dataset(ds_filepath, max_seq_len=self.max_seq_len))

        # get combined set of word types from all datasets
        comb_word_types = []
        for ds in ds_acc:
            comb_word_types.extend(ds.word_types)
        comb_word_types = list(set(comb_word_types))

        # compute word to index mappings that will be shared across datasets
        # pad of 1 accounts for the sequence pad (of 0) down the pipeline
        shared_word_type_to_idx = Dataset.sequence_2_idx(comb_word_types, pad=1)

        # load all the datasets
        for ds in ds_acc:
            ds.load_dataset(shared_word_type_to_idx=shared_word_type_to_idx)

        return ds_acc

    def _load_token_embeddings(self):
        """ Coordinates the loading of pre-trained token embeddings.

        Coordinates the loading of pre-trained token embeddings by reading in
        the file containing the token embeddings and created an embedding matrix
        whos ith row corresponds to the token embedding for the ith word in the
        models word to idx mapping.
        """
        start_time = time.time()
        print('Loading embeddings... ', end='', flush=True)

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

        with open(self.token_pretrained_embedding_filepath, 'r') as pte:
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

    def _setup_model_checkpointing(self):
        """ Sets up per epoch model checkpointing.

        Sets up model checkpointing by:
            1) creating the output_folder if it does not already exists
            2) creating the checkpointing CallBack Keras object

        Returns:
            checkpointer: a Keras CallBack object for per epoch model
            checkpointing
        """
        # create output directory if it does not exist
        make_dir(self.output_folder)
        # create path to output folder
        output_folder_ = os.path.join(self.output_folder,
                                      'model_checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')
        # set up model checkpointing
        checkpointer = ModelCheckpoint(filepath=output_folder_)

        return checkpointer
