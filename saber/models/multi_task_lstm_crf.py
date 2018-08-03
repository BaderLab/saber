from keras import initializers
import keras.backend as K
from keras_contrib.layers.crf import CRF
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import Input
from keras.layers import Lambda
from keras.layers import LSTM
from keras.models import Model
from keras.layers import SpatialDropout1D
from keras.layers import TimeDistributed
from keras.utils import multi_gpu_model
import tensorflow as tf

from ..utils import model_utils

# https://stackoverflow.com/questions/48615003/multi-task-learning-in-keras
# https://medium.com/@literallywords/stratified-k-fold-with-keras-e57c487b1416

# TODO (johngiorgi): I need to stratify the K-folds, but sklearns implementation
# wont handle a y matrix of three dimensions, solve this!
# TODO (johngiorgi): It might be best that the outer most loop (k_folds) just
# generate test/train indicies directly
# TODO (johngiorgi): https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# TODO (johngiorgi): https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

NUM_UNITS_WORD_LSTM = 200
NUM_UNITS_CHAR_LSTM = 200
NUM_UNITS_DENSE = NUM_UNITS_WORD_LSTM // 2
# Note, it appears that only implementation 1 allows for proper variational
# dropout. See: https://keras.io/layers/recurrent/#lstm
IMPLEMENTATION = 1

class MultiTaskLSTMCRF(object):
    """A Keras implementation of a BiLSTM-CRF for sequence labeling."""

    def __init__(self, config, ds, token_embedding_matrix=None):
        # config contains a dictionary of hyperparameters
        self.config = config

        # dataset(s) tied to this instance
        self.ds = ds
        # token embeddings tied to this instance
        self.token_embedding_matrix = token_embedding_matrix
        # model(s) tied to this instance
        self.model = []

    def specify_(self):
        """Specifies a multi-task BiLSTM-CRF for sequence tagging using Keras.

        Implements a hybrid long short-term memory network-condition random
        field (LSTM-CRF) multi-task model for sequence tagging.

        Returns:
            model: a list of keras models, all sharing some number of layers.
        """
        # Specify any shared layers outside the for loop
        # Word-level embedding layer
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

        # Character-level embedding layer
        char_embeddings = Embedding(input_dim=len(self.ds[0].type_to_idx['char']),
                                    output_dim=self.config.char_embed_dim,
                                    mask_zero=True,
                                    name="char_embedding_layer")

        # Char-level BiLSTM
        char_BiLSTM = Bidirectional(LSTM(units=NUM_UNITS_CHAR_LSTM // 2,
                                         return_sequences=False,
                                         dropout=self.config.dropout_rate['input'],
                                         recurrent_dropout=self.config.dropout_rate['recurrent'],
                                         implementation=IMPLEMENTATION),
                                    name="char_BiLSTM")

        # Word-level BiLSTM
        word_BiLSTM_1 = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent'],
                                           implementation=IMPLEMENTATION),
                                      name="word_BiLSTM_1")

        word_BiLSTM_2 = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                           return_sequences=True,
                                           dropout=self.config.dropout_rate['input'],
                                           recurrent_dropout=self.config.dropout_rate['recurrent'],
                                           implementation=IMPLEMENTATION),
                                      name="word_BiLSTM_2")

        # get all unique tag types across all datasets
        all_tag_types = [ds.type_to_idx['tag'] for ds in self.ds]
        all_tag_types = set(x for l in all_tag_types for x in l)

        # Feedforward before CRF, maps each time step to a vector
        feedforward_map = TimeDistributed(Dense(
            units=len(all_tag_types),
            activation=self.config.activation,
            # if activation function is relu, initialize bias to small constant
            # value to avoid dead neurons
            bias_initializer=initializers.Constant(value=0.01) if \
                self.config.activation == 'relu' else 'zeros'), name='feedforward_map')

        # Specify model, taking into account the shared layers
        for ds in self.ds:
            # Word-level embedding.
            word_ids = Input(shape=(None, ), name='word_id_inputs', dtype='int32')
            word_embeddings_shared = word_embeddings(word_ids)

            # Character-level embedding
            char_ids = Input(shape=(None, None), name='char_id_inputs', dtype='int32')
            char_embeddings_shared = char_embeddings(char_ids)
            s = K.shape(char_embeddings_shared)
            # Shape = (batch size, max sentence length, char embedding dimension)
            char_emb_shape = (-1, s[-2], self.config.char_embed_dim)
            char_embeddings_shared = Lambda(lambda x: K.reshape(x, shape=char_emb_shape))(char_embeddings_shared)

            # Character-level BiLSTM + dropout. Spatial dropout applies the
            # same dropout mask to all timesteps which is necessary to implement
            # variational dropout (https://arxiv.org/pdf/1512.05287.pdf)
            char_embeddings_shared = char_BiLSTM(char_embeddings_shared)
            # Shape = (batch size, max sentence length, char BiLSTM hidden size)
            char_lstm_shape = (-1, s[1], NUM_UNITS_CHAR_LSTM)
            char_embeddings_shared = Lambda(lambda x: K.reshape(x, shape=char_lstm_shape))(char_embeddings_shared)
            if self.config.variational_dropout:
                print(' using variational dropout...', end=' ')
                char_embeddings_shared = SpatialDropout1D(self.config.dropout_rate['output'])(char_embeddings_shared)

            # Concatenate word- and char-level embeddings + dropout
            model = Concatenate(axis=-1)([word_embeddings_shared, char_embeddings_shared])
            model = Dropout(self.config.dropout_rate['output'])(model)

            # Word-level BiLSTM + dropout
            model = word_BiLSTM_1(model)
            if self.config.variational_dropout:
                model = SpatialDropout1D(self.config.dropout_rate['output'])(model)

            model = word_BiLSTM_2(model)
            if self.config.variational_dropout:
                model = SpatialDropout1D(self.config.dropout_rate['output'])(model)

            # Feedforward before CRF
            model = feedforward_map(model)

            # CRF output layer
            crf = CRF(len(ds.type_to_idx['tag']), name='crf_classifier')
            output_layer = crf(model)

            # Fully specified model.
            # Instantiate the base model (or "template" model). We recommend doing this with under
            # a CPU device scope, so that the model's weights are hosted on CPU memory. Otherwise
            # they may end up hosted on a GPU, which would complicate weight sharing.
            # https://github.com/keras-team/keras/blob/bf1378f39d02b7d0b53ece5458f9275ac8208046/keras/utils/multi_gpu_utils.py
            # if self.config.gpus >= 2:
            with tf.device('/cpu:0'):
                model = Model(inputs=[word_ids, char_ids], outputs=[output_layer])
            # Else, allow Keras to decide where to put model weights
            # else:
            #    model = Model(inputs=[word_ids, char_ids], outputs=[output_layer])
            self.model.append(model)

        return self.model

    def compile_(self):
        """Compiles a bi-directional multi-task LSTM-CRF for sequence tagging using Keras."""
        for i in range(len(self.model)):
            # Parallize the model if multiple GPUs are available
            # https://github.com/keras-team/keras/pull/9226
            crf_loss_function = self.model[i].layers[-1].loss_function

            try:
                self.model[i] = multi_gpu_model(self.model[i])
                print('using multiple GPUs...', end=' ')
            except:
                print('using single CPU or GPU...', end=' ')

            # need to grab the loss function from models CRF instance
            model_utils.compile_model(model=self.model[i],
                                      loss_function=crf_loss_function,
                                      optimizer=self.config.optimizer,
                                      lr=self.config.learning_rate,
                                      decay=self.config.decay,
                                      clipnorm=self.config.grad_norm)
