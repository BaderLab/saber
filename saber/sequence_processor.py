"""Contains the SequenceProcessor class, which exposes most of Sabers functionality.
"""
from itertools import chain
import logging
import pickle
from pprint import pprint
import os
import time

from gensim.models import KeyedVectors
import numpy as np
from spacy import displacy

from . import constants
from .config import Config
from .dataset import Dataset
from .preprocessor import Preprocessor
from .trainer import Trainer
from .utils import generic_utils
from .utils import model_utils

class SequenceProcessor(object):
    """A class for handeling the loading, saving and training of sequence models.

    Args:
        config (Config): A Config object which contains a set of harmonzied arguments provided in
            a .ini file and, optionally, from the command line. If not provided, a new instance of
            Config is used.
    """
    def __init__(self, config=None):
        self.log = logging.getLogger(__name__)

        # hyperparameters
        self.config = config if config is not None else Config()

        # dataset(s) tied to this instance
        self.ds = []
        # token embeddings tied to this instance
        self.token_embedding_matrix = None

        # model object tied to this instance
        self.model = None

        # preprocessor
        self.preprocessor = Preprocessor()

        if self.config.verbose:
            print('Hyperparameters and model details:')
            pprint(self.config.args)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    def annotate(self, text, model_idx=0, jupyter=False):
        """Uses a trained model to annotate `text`, returns the results in a dictionary.

        For the model at self.model.model[model_idx], coordinates a prediction step on `text`.
        Returns a dictionary containing the cleaned `text` (`text`), and any annotations made by the
        model (`ents`). This dictionary can easily be converted to a json. Optionally,
        renders and HTMl visilization of the annotations made by the model, for use in a jupyter
        notebook.

        text (str): raw text to annotate
        model_idx (int): index of model to use for prediction, defaults to 0
        jupyter (bool): if True, annotations made by the model are rendered in HTML, which can be
            visualized in a jupter notebook.

        Returns:
            dictionary containing the processed input text ('text') and any annotations made by the
            model ('ents').

        Raises:
            ValueError if `text` is invalid (not a string, or empty/falsey).
        """
        if not isinstance(text, str) or not text:
            err_msg = "Argument 'text' must be a valid, non-empty string!"
            self.log.error("ValueError: %s", err_msg)
            raise ValueError(err_msg)

        # model and its corresponding dataset
        ds = self.ds[model_idx]
        model = self.model.model[model_idx]

        # process raw input text, collect input to ML model
        transformed_text = self.transform(text, model_idx)
        model_input = [transformed_text['word2idx'], transformed_text['char2idx']]

        # perform prediction, convert from one-hot to predicted indices, flatten results
        y_pred = model.predict(model_input, batch_size=constants.PRED_BATCH_SIZE)
        y_pred = np.asarray(y_pred.argmax(-1)).ravel()
        # convert predictions to tags and chunk
        pred_tag_seq = [ds.idx_to_tag[idx] for idx in y_pred if ds.idx_to_tag[idx] != constants.PAD]
        pred_chunk_seq = self.preprocessor.chunk_entities(pred_tag_seq)
        # flatten the token offsets
        offsets = list(chain.from_iterable(transformed_text['offsets']))

        # accumulate for predicted entities
        ents = []
        for chunk in pred_chunk_seq:
            # create the entity
            # chunks look like (label, start, end)
            start, end = offsets[chunk[1]][0], offsets[chunk[-1] - 1][-1]
            ents.append({'start': start,
                         'end': end,
                         'text': transformed_text['text'][start:end],
                         'label': chunk[0]})

        annotation = {
            'text': transformed_text['text'],
            'ents': ents,
            'title': None
        }

        if jupyter:
            displacy.render(annotation, jupyter=jupyter, style='ent', manual=True,
                            options=constants.OPTIONS)

        return annotation

    def transform(self, text, ds_idx=0):
        """Processes raw text, returns a dictionary of useful values.

        For the given raw text, returns a dictionary containing the following:
            - 'text': raw text, with minimal processing
            - 'sentences': a list of lists, contains the tokens in each sentence
            - 'offsets': A list of list of tuples containing the start and end
                indices of every token in 'text'
            - 'word2idx': 2-D numpy array containing the token index of every
                token in 'text'. Index is chosen based on the mapping of
                self.ds[model]
            - 'char2idx': 3-D numpy array containing the character index of
                every character in 'text'. Index is chosen based on the mapping
                of self.ds[model]

        Args:
            text (str): raw text to process
            ds_idx (int): which ds (in list self.ds) to use for the mapping of
                token and character indices.
        """
        ds = self.ds[ds_idx]
        return self.preprocessor.transform(text, ds.type_to_idx['word'], ds.type_to_idx['char'])

    def save(self, filepath=None, compress=True, model=0):
        """Coordinates the saving of Saber models.

        Saves the necessary files for model persistance to filepath. Filepath defaults to
        "`self.config.output_folder`/pretrained_models/dataset_names"

        Args:
            filepath (str): directory path to save model folder, defaults to
                "`self.config.output_folder`/pretrained_models/dataset_names"
            compress (bool): True if model should be saved as tarball
            model (int): which model in self.model.model to save, defaults to 0

        Returns:
            True if model was saved without error.
        """
        if filepath is None:
            filepath = generic_utils.get_pretrained_model_dir(self.config)

        generic_utils.make_dir(filepath)
        # create filepaths
        weights_filepath = os.path.join(filepath, 'model_weights.hdf5')
        attributes_filepath = os.path.join(filepath, 'model_attributes.pickle')

        # create a dictionary containg everything we need to save the model
        model_attributes = {'embedding_matrix':self.token_embedding_matrix,
                            'word_embedding_dim': self.config.word_embed_dim,
                            'char_embedding_dim': self.config.char_embed_dim,
                            'type_to_idx': self.ds[model].type_to_idx,
                            'idx_to_tag': self.ds[model].idx_to_tag,}
        # save weights
        self.model.model[model].save_weights(weights_filepath)
        # save attributes
        pickle.dump(model_attributes, open(attributes_filepath, 'wb'))
        # save config file
        self.config.save(filepath)

        if compress:
            generic_utils.compress_model(filepath)

        print('Model saved to {}'.format(filepath))
        self.log.info('Model was saved to %s', filepath)
        return True

    def load(self, filepath):
        """Coordinates the loading of Saber models.

        Loads the necessary files for model creation from filepath. Creates and compiles a
        Keras model based on these files.

        Args:
            filepath (str): directory path to saved pre-trained model folder
        """
        generic_utils.decompress_model(filepath)

        # load attributes, these attributes must be carried over from saved model
        weights_filepath = os.path.join(filepath, 'model_weights.hdf5')
        attributes_filepath = os.path.join(filepath, 'model_attributes.pickle')
        model_attributes = pickle.load(open(attributes_filepath, "rb"))
        self.config.word_embed_dim = model_attributes['word_embedding_dim']
        self.config.char_embed_dim = model_attributes['char_embedding_dim']
        self.token_embedding_matrix = model_attributes['embedding_matrix']

        # create a new dataset instance, load the required attributes for model prediction
        # TEMP: this is an ugly hack, need way around having to provide a filepath
        dummy_ds = os.path.abspath('saber/tests/resources/dummy_dataset_1')
        self.ds = [Dataset(dummy_ds)]
        self.ds[0].type_to_idx = model_attributes['type_to_idx']
        self.ds[0].idx_to_tag = model_attributes['idx_to_tag']

        # specify model based on saved models attributes, load weights and compile
        self.create_model(compile_model=False)
        # by_name loads allows us to load a model even if when architecture has changed slightly
        self.model.model[0].load_weights(weights_filepath, by_name=True)
        self.model.compile_()

        return self

    def load_dataset(self):
        """Coordinates the loading of a dataset."""
        start = time.time()
        if not self.config.dataset_folder:
            err_msg = "Must provide at least one dataset via the 'dataset_folder' parameter"
            self.log.error('AssertionError %s', err_msg)
            raise AssertionError(err_msg)

        # if not None, then pre-trained model has been loaded, use its type mapping
        type_to_idx = None if not self.ds else self.ds[0].type_to_idx
        # Datasets may be 'single' or 'compound' (more than one). Consider a dataset single if
        # there is only one filepath in self.config.dataset_folder' and compound otherwise.
        if len(self.config.dataset_folder) == 1:
            print('Loading (single) dataset... ', end='', flush=True)
            self.ds = self._load_single_dataset(type_to_idx)
            self.log.info('Loaded single dataset at: %s', self.config.dataset_folder)
        else:
            print('Loading (compound) dataset... ', end='', flush=True)
            self.ds = self._load_compound_dataset(type_to_idx)
            self.log.info('Loaded multiple datasets at: %s', self.config.dataset_folder)

        end = time.time() - start
        print('Done ({0:.2f} seconds).'.format(end))

        return self

    def _load_single_dataset(self, type_to_idx=None):
        """Loads a single dataset.

        Creates and loads a single dataset object for a dataset at self.config.dataset_folder[0].

        Args:
            type_to_idx (dict): a mapping of types ('word', 'char') to unique integer ids, when
                provided, these are used in the loading of the dataset at
                `self.config.dataset_folder[0]`

        Returns:
            a list containing a single dataset object.
        """
        ds = Dataset(self.config.dataset_folder[0],
                     replace_rare_tokens=self.config.replace_rare_tokens)
        if type_to_idx is not None:
            ds.load_data_and_labels()
            ds.get_types()

        ds.load_dataset(type_to_idx)

        return [ds]

    def _load_compound_dataset(self, type_to_idx):
        """Loads a compound dataset.

        Creates and loads a 'compound' dataset. Compound datasets are specified by multiple
        individual datasets, and share multiple attributes (such as 'word' and 'char' type to index
        mappings). Loads such a dataset for each dataset at self.dataset_folder.

        Args:
            type_to_idx (dict): a mapping of types ('word', 'char') to unique integer ids, when
                provided, these are used in the loading of the dataset at
                `self.config.dataset_folder[0]`

        Returns:
            A list containing multiple compound dataset objects.
        """
        # accumulate datasets
        compound_ds = [Dataset(ds, replace_rare_tokens=self.config.replace_rare_tokens) for ds in
                       self.config.dataset_folder]

        for ds in compound_ds:
            ds.load_data_and_labels()
            ds.get_types()

        if type_to_idx is None:
            # get combined set of word and char types from all datasets
            combined_types = {'word': [ds.types['word'] for ds in compound_ds],
                              'char': [ds.types['char'] for ds in compound_ds]}
            combined_types['word'] = list(set(chain.from_iterable(combined_types['word'])))
            combined_types['char'] = list(set(chain.from_iterable(combined_types['char'])))

            # compute word to index mappings that will be shared across datasets
            type_to_idx = {'word': Preprocessor.type_to_idx(combined_types['word'],
                                                            constants.INITIAL_MAPPING['word']),
                           'char': Preprocessor.type_to_idx(combined_types['char'],
                                                            constants.INITIAL_MAPPING['word'])}
        # finally, load all the datasets, providing pre-populated type_to_idx mappings
        for ds in compound_ds:
            ds.load_dataset(type_to_idx)

        return compound_ds

    def load_embeddings(self, binary=True):
        """Coordinates the loading of pre-trained token embeddings.

        Args:
            binary (bool): True if pre-trained embeddings are in C binary format, False if they are
                in C text format.

        Raises:
            MissingStepException: if no dataset has been loaded.
            ValueError: If 'self.config.pretrained_embeddings' is None.
        """
        if not self.ds:
            err_msg = "You need to call 'load_dataset()' before calling 'load_embeddings()'"
            self.log.error('MissingStepException: %s', err_msg)
            raise MissingStepException(err_msg)
        if not self.config.pretrained_embeddings:
            err_msg = "'pretrained_embeddings' argument was empty'"
            self.log.error('ValueError: %s', err_msg)
            raise ValueError(err_msg)

        self._load_token_embeddings(binary)

        return self

    def create_model(self, compile_model=True):
        """Specifies and compiles chosen model (self.config.model_name).

        For a chosen model (provided at the command line or in the configuration file and saved as
        `self.config.model_name`), load this models class class. Then 'specify' and 'compile'
        the Keras model(s) it contains.

        Raises:
            ValueError if model name at `self.config.model_name` is not valid
        """
        if self.config.model_name not in ['mt-lstm-crf']:
            err_msg = "Model name is not valid. Check the argument value for 'model_name'"
            self.log.error('ValueError: %s ', err_msg)
            raise ValueError(err_msg)

        start_time = time.time()
        # setup the chosen model
        if self.config.model_name == 'mt-lstm-crf':
            print('Building the multi-task BiLSTM-CRF model... ', end='', flush=True)
            from .models.multi_task_lstm_crf import MultiTaskLSTMCRF
            model = MultiTaskLSTMCRF(config=self.config, ds=self.ds,
                                     token_embedding_matrix=self.token_embedding_matrix)

        # specify and compile the chosen model
        model.specify_()
        if compile_model:
            model.compile_()

        # update this objects model attribute with instance of model class
        self.model = model

        elapsed_time = time.time() - start_time
        print('Done ({0:.2f} seconds).'.format(elapsed_time))
        self.log.info('%s model was built successfully', self.config.model_name.upper())

        if self.config.verbose:
            for i, model in enumerate(self.model.model):
                ds_name = os.path.basename(self.config.dataset_folder[i])
                print('Model architecture for dataset {}:'.format(ds_name))
                model.summary()

        return self

    def fit(self):
        """Fit the specified model.

        For the given model(s) (self.model), sets up per epoch checkpointing and fits the model.

        Returns:
            a list, containing one or more model instances (subclass of
            BaseModel) with trained Keras models.
        """
        # setup callbacks
        callbacks = {'checkpoint': None, 'tensorboard': None}
        train_session_dir = model_utils.prepare_output_directory(self.config.dataset_folder,
                                                                 self.config.output_folder,
                                                                 self.config)
        # model checkpointing
        callbacks['checkpoint'] = model_utils.setup_checkpoint_callback(train_session_dir)
        # tensorboard
        if self.config.tensorboard:
            callbacks['tensorboard'] = model_utils.setup_tensorboard_callback(train_session_dir)

        trainer = Trainer(self.config, self.ds, self.model)
        trainer.train(callbacks, train_session_dir)

        return self

        # fit
        # train_history = self.model.fit_(checkpointer=checkpointer)
        # don't get history for now
        # self.model.fit_(checkpointer, train_session_dir)
        # train_history = pd.DataFrame(train_history.history)
        # return train_history

    def _load_token_embeddings(self, binary=True):
        """Coordinates the loading of pre-trained token embeddings.

        Coordinates the loading of pre-trained token embeddings by reading in the file containing
        the token embeddings and creating a embedding matrix whos ith row corresponds to the token
        embedding for the ith word in the models word to idx mapping.

        Args:
            binary (bool): True if pre-trained embeddings are in C binary format, False if they are
                in C text format.
        """
        start = time.time()
        print('Loading embeddings... ', end='', flush=True)

        # prepare the embedding indicies
        embedding_idx = self._prepare_token_embedding_layer(binary)
        embedding_dim = len(list(embedding_idx.values())[0])
        # create the embedding matrix, update attribute
        embedding_matrix = self._prepare_token_embedding_matrix(embedding_idx, embedding_dim)
        self.token_embedding_matrix = embedding_matrix

        end = time.time() - start
        print('Done ({0:.2f} seconds)'.format(end))
        print('Found {} word vectors of dimension {}'.format(len(embedding_idx), embedding_dim))
        self.log.info('Loaded %i word vectors of dimension %i', len(embedding_idx), embedding_dim)

    def _prepare_token_embedding_layer(self, binary=True):
        """Creates an embedding index using pretrained token embeddings.

        For the pretrained word embeddings given at `self.config.pretrained_embeddings`, creates
        and returns a dictionary mapping words to embeddings, or word vectors. Note that if
        `self.config.debug` is True, only the first 10K vectors are loaded.

        Args:
            binary (bool): True if pre-trained embeddings are in C binary format, False if they are
                in C text format.

        Returns:
            embed_idx (dict): mapping of words to pre-trained word embeddings
        """
        limit = 10000 if self.config.debug else None
        vectors = KeyedVectors.load_word2vec_format(self.config.pretrained_embeddings,
                                                    binary=binary,
                                                    limit=limit)
        embed_idx = {word: vectors[word] for word in vectors.vocab}
        return embed_idx

    def _prepare_token_embedding_matrix(self, embedding_idx, embedding_dim):
        """Creates an embedding matrix using pretrained token embeddings.

        For the models word to index mappings, and word to pre-trained token embeddings, creates a
        matrix which maps all words in the models dataset to a pre-trained token embedding. If the
        token embedding does not exist in the pre-trained token embeddings file, the word will be
        mapped to an embedding of all zeros.

        Args:
            embedding_idx (dict): dictionary mapping words to their dense embeddings
            embedding_size (int): dimension of dense embeddings

        Returns:
            matrix whos ith row corresponds to the word embedding for the ith word in the models
            word to idx mapping.
        """
        # initialize the embeddings matrix
        word_types = len(self.ds[0].type_to_idx['word'])
        token_embedding_matrix = np.zeros((word_types, embedding_dim))

        # lookup embeddings for every word in the dataset
        for word, i in self.ds[0].type_to_idx['word'].items():
            token_embedding = embedding_idx.get(word)
            if token_embedding is not None:
                # words not found in embedding index will be all-zeros.
                token_embedding_matrix[i] = token_embedding

        return token_embedding_matrix

# https://stackoverflow.com/questions/1319615/proper-way-to-declare-custom-exceptions-in-modern-python
class MissingStepException(Exception):
    """Execption subclass for signalling to user that some required previous step was missed."""
    pass
