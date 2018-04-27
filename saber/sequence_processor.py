import os
import time
import json
import pickle
from pprint import pprint
from itertools import chain

import numpy as np
import pandas as pd

from spacy import displacy
from keras.models import load_model
from keras_contrib.layers.crf import CRF

import constants
from dataset import Dataset
from metrics import Metrics
from preprocessor import Preprocessor
from utils_generic import make_dir
from utils_models import create_train_session_dir
from utils_models import setup_model_checkpointing

print('Saber version: {0}'.format('0.1-dev'))

# TODO (johngiorgi): READ: https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/
# TODO (johngiorgi): make model checkpointing a config param
# TODO (johngiorgi): make a debug mode that doesn't load token embeddings and
# loads only some lines of dataset
# TODO (johngiorgi): implement saving loading of models
# TODO (johngiorgi): predict should be more of an interface, calling it should
# return a nicely formatted representation of the prediticted entities.
# TODO (johngiorgi): use proper error handeling for load_ds / load_token methods

class SequenceProcessor(object):
    """A class for handeling the loading, saving, training, and specifying of
    sequence processing models."""

    def __init__(self, config):
        # hyperparameters
        self.config = config

        # dataset(s) tied to this instance
        self.ds = []
        # token embeddings tied to this instance
        self.token_embedding_matrix = None

        # model object tied to this instance
        self.model = None

        # preprocessor
        self.preprocessor = Preprocessor()

        if self.config['verbose']: pprint(self.config)

    def predict(self, text, model=0, jupyter=False, *args, **kwargs):
        """Performs prediction for a given model and returns results."""
        ds_ = self.ds[model]
        model_ = self.model.model[model]

        # get reverse mapping of indices to tags
        # TODO: change to the following statement with new models
        # idx2tag = ds_.idx_to_tag_type
        idx2tag = {v: k for k, v in ds_.tag_type_to_idx.items()}
        # process raw input text
        transformed_text = self.preprocessor.transform(text,
                                                       ds_.word_type_to_idx,
                                                       ds_.char_type_to_idx)

        # perform prediction, convert to tag sequence
        y_pred = model_.predict([transformed_text['word_idx_sequence'],
                                 transformed_text['char_idx_sequence']]).argmax(-1)
        idx_pred_seq = np.asarray(y_pred).ravel()
        # TODO: clean this up, need to drop pads.
        tag_pred_seq = [idx2tag[idx] for idx in idx_pred_seq if idx2tag[idx] != '<PAD>']

        # chunk the predicted entities
        chunk_pred_seq = self.preprocessor.chunk_entities(tag_pred_seq)
        # flatten the token offsets
        offsets = list(chain.from_iterable(transformed_text['offsets']))

        # TODO: this should be moved to a unit test
        assert len(tag_pred_seq) == len(offsets)

        # accumulator for predicted entities
        ents = []

        for chunk in chunk_pred_seq:
            # get token indicies of the labeled chunk
            chunk_start = chunk[1]
            chunk_end = chunk[-1] - 1
            # character indicies of the labeled chunk
            start, end = offsets[chunk_start][0], offsets[chunk_end][-1]
            # create the entity
            ents.append({'start': start,
                         'end': end,
                         'label': chunk[0]})

        annotation = {
            'text': transformed_text['text'],
            'ents': ents,
            'title': None
        }

        if jupyter:
            displacy.render(annotation, jupyter=jupyter, style='ent',
                            manual=True, options=constants.OPTIONS)

        return json.dumps(annotation, ensure_ascii=False)

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath, model=0):
        """Coordinates the saving of Saber models.

        Saves the necessary files for model persistance to filepath.

        Args:
            filepath (str): directory path to save model folder to
            model (int): which model in self.model.model to save, defaults to 0
        """
        # create the pretrained model folder
        make_dir(os.path.join(filepath))

        # create a dictionary containg everything we need to save the model
        model_attributes = {}

        model_attributes['config'] = self.config
        model_attributes['token_embeddings'] = self.token_embedding_matrix
        model_attributes['ds'] = self.ds[model]

        # create filepaths
        weights_filepath = os.path.join(filepath, 'model_weights.hdf5')
        attributes_filepath = os.path.join(filepath, 'model_attributes.pickle')

        # save weights
        self.model.model[model].save_weights(weights_filepath)
        # save attributes
        pickle.dump(model_attributes, open(attributes_filepath, 'wb'))

    def load(self, filepath):
        """Coordinates the saving of Saber models.

        Loads the necessary files for model creation from filepath.

        Args:
            filepath (str): directory path to saved pretrained folder
        """
        # create filepaths
        weights_filepath = os.path.join(filepath, 'model_weights.hdf5')
        attributes_filepath = os.path.join(filepath, 'model_attributes.pickle')

        # load attributes
        model_attributes = pickle.load(open(attributes_filepath, "rb" ))
        self.config = model_attributes['config']
        self.ds = [model_attributes['ds']]
        self.token_embedding_matrix = model_attributes['token_embeddings']

        # create model based on saved models attributes
        self.create_model()

        # load weights
        self.model.model[0].load_weights(weights_filepath)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def load_dataset(self):
        """Coordinates the loading of a dataset."""
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
        print('Done ({0:.2f} seconds).'.format(elapsed_time))

    def load_embeddings(self):
        """Coordinates the loading of pre-trained token embeddings."""
        assert self.ds, 'You must load a dataset before loading token embeddings'
        assert self.config['token_pretrained_embedding_filepath'] is not None, 'Token embedding filepath must be provided in the config file or at the command line'

        self._load_token_embeddings()

    def create_model(self):
        """Specifies and compiles chosen model (self.config['model_name'])."""
        assert self.config['model_name'] in ['MT-LSTM-CRF'], 'Model name is not valid.'

        start_time = time.time()
        # setup the chosen model
        if self.config['model_name'] == 'MT-LSTM-CRF':
            print('[INFO] Building the multi-task LSTM-CRF model... ', end='', flush=True)
            from models.multi_task_lstm_crf import MultiTaskLSTMCRF
            model_ = MultiTaskLSTMCRF(config=self.config,
                                      ds=self.ds,
                                      token_embedding_matrix=self.token_embedding_matrix)

        # specify and compile the chosen model
        model_.specify_()
        model_.compile_()
        # update this objects model attribute with instance of model class
        self.model = model_

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds).'.format(elapsed_time))

    def fit(self):
        """Fit the specified model.

        For the given model (self.model), sets up per epoch checkpointing
        and fits the model.

        Returns:
            train_hist, the history of the model training as a pandas
            dataframe.
        """
        # setup model checkpointing
        train_session_dir = create_train_session_dir(self.config['dataset_folder'],
                                                     self.config['output_folder'])
        checkpointer = setup_model_checkpointing(train_session_dir)

        # fit
        # train_history = self.model.fit_(checkpointer=checkpointer)
        # don't get history for now
        self.model.fit_(checkpointer, train_session_dir)
        # train_history = pd.DataFrame(train_history.history)
        # return train_history

    def _load_single_dataset(self):
        """Loads a single dataset.

        Creates and loads a single dataset object for a dataset at
        self.dataset_folder[0].

        Returns:
            a list containing a single dataset object.
        """
        ds = Dataset(self.config['dataset_folder'][0])
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
        # accumulate datasets
        compound_ds = [Dataset(ds) for ds in self.config['dataset_folder']]

        for ds in compound_ds:
            ds.load_data_and_labels()
            ds.get_types()

        # get combined set of word types from all datasets
        comb_word_types = []
        comb_char_types = []
        for ds in compound_ds:
            comb_word_types.extend(ds.word_types)
            comb_char_types.extend(ds.char_types)
        comb_word_types = list(set(comb_word_types))
        comb_char_types = list(set(comb_char_types))

        # compute word to index mappings that will be shared across datasets
        # pad of 1 accounts for the sequence pad (of 0) down the pipeline
        word_type_to_idx = Preprocessor.sequence_to_idx(comb_word_types)
        char_type_to_idx = Preprocessor.sequence_to_idx(comb_char_types)

        # load all the datasets
        for ds in compound_ds:
            ds.load_dataset(word_type_to_idx, char_type_to_idx)

        return compound_ds

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
        embedding_index = self._prepare_token_embedding_layer()
        embedding_dimension = len(list(embedding_index.values())[0])
        # create the embedding matrix, update attribute
        embedding_matrix = self._prepare_token_embedding_matrix(embedding_index, embedding_dimension)
        self.token_embedding_matrix = embedding_matrix

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds)'.format(elapsed_time))
        print('{s}Found {t} word vectors of dimension {d}'.format(
            s=' ' * 7,
            t=len(embedding_index),
            d=embedding_dimension))

    def _prepare_token_embedding_layer(self):
        """Creates an embedding index using pretrained token embeddings.

        For the models given pretrained token embeddings, creates and returns a
        dictionary mapping words to known embeddings.

        Returns:
            embedding_index: mapping of words to pre-trained token embeddings
        """
        # acc
        embedding_index = {}

        # open pre-trained token embedding file for reading
        with open(self.config['token_pretrained_embedding_filepath'], 'r') as pte:
            for line in pte:
                # split line, get word and its embedding
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')

                # update our embedding index
                embedding_index[word] = coefs

        return embedding_index

    def _prepare_token_embedding_matrix(self,
                                        embedding_index,
                                        embedding_size):
        """Creates an embedding matrix using pretrained token embeddings.

        For the models word to index mappings, and word to pre-trained token
        embeddings, creates a matrix which maps all words in the models dataset
        to a pre-trained token embedding. If the token embedding does not exist
        in the pre-trained token embeddings file, the word will be mapped to
        an embedding of all zeros.

        Returns:
            token_embedding_matrix: a matrix whos ith row corresponds to the
            token embedding for the ith word in the models word to idx mapping.
        """
        # initialize the embeddings matrix
        token_embedding_matrix = np.zeros((len(self.ds[0].word_type_to_idx),
                                           embedding_size))

        # lookup embeddings for every word in the dataset
        for word, i in self.ds[0].word_type_to_idx.items():
            token_embedding = embedding_index.get(word)
            if token_embedding is not None:
                # words not found in embedding index will be all-zeros.
                token_embedding_matrix[i] = token_embedding

        return token_embedding_matrix
