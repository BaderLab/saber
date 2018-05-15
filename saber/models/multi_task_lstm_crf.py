import os
import time
from operator import itemgetter

import numpy as np

import keras.backend as K
from keras import optimizers
from keras.layers import LSTM
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras_contrib.layers.crf import CRF

import utils_models
from metrics import Metrics

# https://stackoverflow.com/questions/48615003/multi-task-learning-in-keras
# https://machinelearningmastery.com/keras-functional-api-deep-learning/
# https://medium.com/@literallywords/stratified-k-fold-with-keras-e57c487b1416

# TODO (johngiorgi): I need to stratify the K-folds, but sklearns implementation
# wont handle a y matrix of three dimensions, solve this!
# TODO (johngiorgi): It might be best that the outer most loop (k_folds) just
# generate test/train indicies directly
# TODO (johngiorgi): https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# TODO (johngiorgi): make sure this process is shuffling the data
# TODO (johngiorgi): https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

NUM_UNITS_WORD_LSTM = 200
NUM_UNITS_CHAR_LSTM = 200
NUM_UNITS_DENSE = NUM_UNITS_WORD_LSTM // 2
# see: https://keras.io/layers/recurrent/#lstm
IMPLEMENTATION = 2

class MultiTaskLSTMCRF(object):
    """A Keras implementation of a BiLSTM-CRF for sequence labeling."""

    def __init__(self, config, ds, token_embedding_matrix=None):
        # config contains a dictionary of hyperparameters
        self.config = config

        # dataset(s) tied to this instance
        self.ds = ds
        # token embeddings tied to this instance
        self.token_embedding_matrix = token_embedding_matrix

        # metric(s) object tied to this instance, one per dataset
        self.metrics = []
        # model(s) tied to this instance
        self.model = []
        self.crf = []

    def specify_(self):
        """Specifies a multi-task bidirectional LSTM-CRF for sequence tagging
        using Keras.

        Implements a hybrid long short-term memory network-condition random
        field (LSTM-CRF) multi-task model for sequence tagging.

        Returns:
            model: a list of keras models, sharing (excluding the crf layer)
                   some number of layers.
            crf: a list of task-specific crf layers implemented using
                 keras.contrib, one for each model.
        """
        # Specify any shared layers outside the for loop

        # Word-level embedding layer
        if self.token_embedding_matrix is None:
            word_embeddings = Embedding(input_dim=len(self.ds[0].word_type_to_idx),
                                        output_dim=self.config['token_embedding_dimension'],
                                        mask_zero=True)
        else:
            word_embeddings = Embedding(input_dim=len(self.ds[0].word_type_to_idx),
                                        output_dim=self.token_embedding_matrix.shape[1],
                                        mask_zero=True,
                                        weights=[self.token_embedding_matrix],
                                        trainable=(not self.config['freeze_token_embeddings']))

        # Character-level embedding layer
        char_embeddings = Embedding(input_dim=(len(self.ds[0].char_type_to_idx)),
                                    output_dim=self.config['character_embedding_dimension'],
                                    mask_zero=True)

        # Char-level BiLSTM
        fwd_state = LSTM(NUM_UNITS_CHAR_LSTM // 2, return_state=True,
                                                   dropout=self.config['dropout_rate'],
                                                   recurrent_dropout=self.config['dropout_rate'])
        bwd_state = LSTM(NUM_UNITS_CHAR_LSTM // 2, return_state=True,
                                                   go_backwards=True,
                                                   dropout=self.config['dropout_rate'],
                                                   recurrent_dropout=self.config['dropout_rate'])



        # Word-level BiLSTM
        word_BiLSTM = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                         return_sequences=True,
                                         dropout=self.config['dropout_rate'],
                                         recurrent_dropout=self.config['dropout_rate']))

        # Feedforward before CRF
        feedforward_af_word_lstm = TimeDistributed(
            Dense(units=NUM_UNITS_DENSE,
                  activation=self.config['activation_function']))


        # get all unique tag types across all datasets
        all_tag_types = [ds.tag_types for ds in self.ds]
        all_tag_types = set(x for l in all_tag_types for x in l)

        feedforward_bf_crf = TimeDistributed(
            Dense(units=len(all_tag_types),
                  activation=self.config['activation_function']))

        # Specify model, taking into account the shared layers
        for ds in self.ds:
            # Word-level embedding layers
            word_ids = Input(shape=(None, ), dtype='int32')
            word_embeddings_shared = word_embeddings(word_ids)

            # Character-level embedding layers
            char_ids = Input(shape=(None, None), dtype='int32')
            char_embeddings_shared = char_embeddings(char_ids)
            s = K.shape(char_embeddings_shared)
            # Shape = (batch size, max sentence length, char embedding dimension)
            char_embeddings_shared = Lambda(lambda x: K.reshape(
                x, shape=(-1, s[-2], self.config['character_embedding_dimension'])))(char_embeddings_shared)

            # Character-level BiLSTM
            fwd_state_shared = fwd_state(char_embeddings_shared)[-2]
            bwd_state_shared = bwd_state(char_embeddings_shared)[-2]
            char_embeddings_shared = Concatenate(axis=-1)([fwd_state_shared, bwd_state_shared])
            # Shape = (batch size, max sentence length, char BiLSTM hidden size)
            char_embeddings_shared = Lambda(lambda x: K.reshape(
                x, shape=(-1, s[1], NUM_UNITS_CHAR_LSTM)))(char_embeddings_shared)

            # Concatenate word- and char-level embeddings + dropout
            model = Concatenate(axis=-1)([word_embeddings_shared, char_embeddings_shared])
            model = Dropout(self.config['dropout_rate'])(model)

            # Word-level BiLSTM + dropout
            model = word_BiLSTM(model)
            model = Dropout(self.config['dropout_rate'])(model)

            # Feedforward after word-level BiLSTM + dropout
            model = feedforward_af_word_lstm(model)
            model = Dropout(self.config['dropout_rate'])(model)
            # Feedforward before CRF + dropout
            model = feedforward_bf_crf(model)
            model = Dropout(self.config['dropout_rate'])(model)

            # CRF output layer
            crf = CRF(len(ds.tag_types))
            output_layer = crf(model)

            # Fully specified model
            self.model.append(Model(inputs=[word_ids, char_ids], outputs=[output_layer]))
            self.crf.append(crf)

        return self.model, self.crf

    def compile_(self):
        """Compiles a bidirectional multi-task LSTM-CRF for sequence tagging
        using Keras."""
        for model, crf in zip(self.model, self.crf):
            utils_models.compile_model(model=model,
                                       loss_function=crf.loss_function,
                                       optimizer=self.config['optimizer'],
                                       lr=self.config['learning_rate'],
                                       decay=self.config['decay'],
                                       clipnorm=self.config['gradient_normalization'],
                                       verbose=self.config['verbose'])

    def fit_(self, checkpointer, output_dir):
        """Fits a bidirectional multi-task LSTM-CRF for for sequence tagging
        using Keras."""
        # get train/valid indicies for each dataset
        train_valid_indices = utils_models.get_train_valid_indices(self.ds, self.config['k_folds'])

        ## FOLDS
        for fold in range(self.config['k_folds']):
            # get the train/valid partitioned data for all datasets
            data_partitions = utils_models.get_data_partitions(self.ds, train_valid_indices, fold)
            # create the Keras Callback object for computing/storing metrics
            metrics_current_fold = utils_models.get_metrics(self.ds, data_partitions, output_dir)
            ## EPOCHS
            for epoch in range(self.config['maximum_number_of_epochs']):
                print('[INFO] Fold: {}/{}; Global epoch: {}/{}'.format(fold + 1,
                                                                       self.config['k_folds'],
                                                                       epoch + 1,
                                                                       self.config['maximum_number_of_epochs']))
                ## DATASETS/MODELS
                for i, (ds, model) in enumerate(zip(self.ds, self.model)):
                    # mainly for cleanliness
                    X_word_train = np.array(data_partitions[i][0])
                    X_word_valid = np.array(data_partitions[i][1])
                    X_char_train = np.array(data_partitions[i][2])
                    X_char_valid = np.array(data_partitions[i][3])
                    y_train = np.array(data_partitions[i][4])
                    y_valid = np.array(data_partitions[i][5])

                    model.fit(x=[X_word_train, X_char_train],
                              y=[y_train],
                              batch_size=self.config['batch_size'],
                              epochs=1,
                              callbacks=[checkpointer[i],
                                         metrics_current_fold[i]],
                              validation_data=([X_word_valid, X_char_valid],
                                               [y_valid]),
                              verbose=1)


            self.metrics.append(metrics_current_fold)

            # end of a k-fold, so clear the model, specify and compile again
            if fold < self.config['k_folds'] - 1:
                self.model = []
                self.crf = []
                self.specify_()
                self.compile_()
