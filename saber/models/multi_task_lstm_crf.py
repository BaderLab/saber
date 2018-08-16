"""Contains the class for the Multi-task BiLSTM-CRF (MT-BILSTM-CRF) Keras model for squence
labelling.
"""
import json
import logging

from keras import initializers
from keras_contrib.layers.crf import CRF
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Input
from keras.layers import LSTM
from keras.models import Model
from keras.models import model_from_json
from keras.layers import SpatialDropout1D
from keras.layers import TimeDistributed
from keras.utils import multi_gpu_model
import tensorflow as tf

from ..utils import model_utils

# TODO (johngiorgi): I should to stratify the K-folds...

NUM_UNITS_WORD_LSTM = 200
NUM_UNITS_CHAR_LSTM = 200
NUM_UNITS_DENSE = NUM_UNITS_WORD_LSTM // 2

class MultiTaskLSTMCRF(object):
    """A Keras implementation of a BiLSTM-CRF for sequence labeling.

    A BiLSTM-CRF for NER implementation in Keras. Supports multi-task learning by default, just pass
    multiple Dataset objects via `ds` to the constructor and the model will share all the parameters
    of all layers, except for the final output layer, across all datasets, where each dataset
    represents a NER task.

    Args:
        config (Config): A Config object which contains a set of harmonzied arguments provided in
            a .ini file and, optionally, from the command line. If not provided, a new instance of
            Config is used.
        ds (list): a list of Dataset objects.
        token_embedding_matrix (numpy.ndarray): a numpy array where ith row contains the vector
            embedding for ith word type.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """
    def __init__(self, config, ds, token_embedding_matrix=None):
        # hyperparameters and model details
        self.config = config

        # dataset(s) tied to this instance
        self.ds = ds
        # token embeddings tied to this instance
        self.token_embedding_matrix = token_embedding_matrix
        # model(s) tied to this instance
        self.model = []

        self.log = logging.getLogger(__name__)

    def save(self, weights_filepath, model_filepath, model=0):
        """Save a model to disk.

        Args:
            weights_filepath (str): filepath to the models wieghts (.hdf5 file)
            model_filepath (str): filepath to the models architecture (.json file)
            model (int): which model from `self.model` to save
        """
        with open(model_filepath, 'w') as f:
            model_json = self.model[model].to_json()
            json.dump(json.loads(model_json), f, sort_keys=True, indent=4)
            self.model[model].save_weights(weights_filepath)

    def load(self, weights_filepath, model_filepath):
        """Load a model from disk.

        Args:
            weights_filepath (str): filepath to the models wieghts (.hdf5 file)
            model_filepath (str): filepath to the models architecture (.json file)
        """
        with open(model_filepath) as f:
            model = model_from_json(f.read(), custom_objects={'CRF': CRF})
            model.load_weights(weights_filepath)
            self.model.append(model)

    def specify_(self):
        """Specifies a multi-task BiLSTM-CRF for sequence tagging using Keras.

        Implements a hybrid long short-term memory network-condition random
        field (LSTM-CRF) multi-task model for sequence tagging.

        Returns:
            model: a list of keras models, all sharing some number of layers.
        """
        # specify any shared layers outside the for loop
        # word-level embedding layer
        if self.token_embedding_matrix is None:
            word_embeddings = Embedding(input_dim=len(self.ds[0].type_to_idx['word']),
                                        output_dim=self.config.word_embed_dim,
                                        mask_zero=True,
                                        name="word_embedding_layer")
        else:
            word_embeddings = Embedding(input_dim=len(self.ds[0].type_to_idx['word']),
                                        output_dim=self.token_embedding_matrix.shape[1],
                                        mask_zero=True,
                                        weights=[self.token_embedding_matrix],
                                        trainable=self.config.fine_tune_word_embeddings,
                                        name="word_embedding_layer")
        # character-level embedding layer
        char_embeddings = Embedding(input_dim=len(self.ds[0].type_to_idx['char']),
                                    output_dim=self.config.char_embed_dim,
                                    mask_zero=True,
                                    name="char_embedding_layer")
        # char-level BiLSTM
        char_BiLSTM = TimeDistributed(Bidirectional(LSTM(NUM_UNITS_CHAR_LSTM // 2)))
        # word-level BiLSTM
        word_BiLSTM_1 = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent']),
                                      name="word_BiLSTM_1")
        word_BiLSTM_2 = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent']),
                                      name="word_BiLSTM_2")

        # get all unique tag types across all datasets
        all_tag_types = [ds.type_to_idx['tag'] for ds in self.ds]
        all_tag_types = set(x for l in all_tag_types for x in l)

        # feedforward before CRF, maps each time step to a vector
        # if activation function relu, initialize bias to small constant to avoid dead neurons
        bias_init = initializers.Constant(value=0.01) if self.config.activation == 'relu' else 'zeros'
        feedforward_map = TimeDistributed(Dense(len(all_tag_types),
                                                activation=self.config.activation,
                                                bias_initializer=bias_init),
                                          name='feedforward_map')
        # specify model, taking into account the shared layers
        for ds in self.ds:
            # word-level embedding
            word_ids = Input(shape=(None, ), dtype='int32', name='word_id_inputs')
            word_embeddings = word_embeddings(word_ids)

            # character-level embedding
            char_ids = Input(shape=(None, None), dtype='int32', name='char_id_inputs')
            char_embeddings = char_embeddings(char_ids)
            # character-level BiLSTM + dropout. Spatial dropout applies the same dropout mask to all
            # timesteps which is necessary to implement variational dropout
            # (https://arxiv.org/pdf/1512.05287.pdf)
            char_embeddings = char_BiLSTM(char_embeddings)
            if self.config.variational_dropout:
                self.log.info('Used variational dropout')
                char_embeddings = SpatialDropout1D(self.config.dropout_rate['output'])(char_embeddings)

            # concatenate word- and char-level embeddings + dropout
            model = Concatenate(axis=-1)([word_embeddings, char_embeddings])
            model = Dropout(self.config.dropout_rate['output'])(model)

            # word-level BiLSTM + dropout
            model = word_BiLSTM_1(model)
            if self.config.variational_dropout:
                model = SpatialDropout1D(self.config.dropout_rate['output'])(model)
            model = word_BiLSTM_2(model)
            if self.config.variational_dropout:
                model = SpatialDropout1D(self.config.dropout_rate['output'])(model)

            # feedforward before CRF
            model = feedforward_map(model)

            # CRF output layer
            crf = CRF(len(ds.type_to_idx['tag']), name='crf_classifier')
            output_layer = crf(model)

            # fully specified model
            # https://github.com/keras-team/keras/blob/bf1378f39d02b7d0b53ece5458f9275ac8208046/keras/utils/multi_gpu_utils.py
            with tf.device('/cpu:0'):
                model = Model(inputs=[word_ids, char_ids], outputs=[output_layer])
            self.model.append(model)

    def compile_(self):
        """Compiles a bi-directional multi-task LSTM-CRF for sequence tagging using Keras."""
        for i in range(len(self.model)):
            # parallize the model if multiple GPUs are available
            # https://github.com/keras-team/keras/pull/9226
            try:
                self.model[i] = multi_gpu_model(self.model[i])
                self.log.info('Compiling the model on multiple GPUs')
            # awfully bad practice but this was the example given by Keras documentation
            except:
                self.log.info('Compiling the model on a single CPU or GPU')

            # need to grab the loss function from models CRF instance
            crf_loss_function = self.model[i].layers[-1].loss_function
            model_utils.compile_model(model=self.model[i],
                                      loss_function=crf_loss_function,
                                      optimizer=self.config.optimizer,
                                      lr=self.config.learning_rate,
                                      decay=self.config.decay,
                                      clipnorm=self.config.grad_norm)
