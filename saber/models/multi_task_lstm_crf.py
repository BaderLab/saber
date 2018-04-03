import os
import time
from time import strftime
from operator import itemgetter

import numpy as np
from sklearn.model_selection import KFold

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

from metrics import Metrics
from utils_generic import make_dir
from utils_models import compile_model

# https://stackoverflow.com/questions/48615003/multi-task-learning-in-keras
# https://machinelearningmastery.com/keras-functional-api-deep-learning/
# https://medium.com/@literallywords/stratified-k-fold-with-keras-e57c487b1416

# TODO (johngiorgi): the way I get train/test partitions is likely copying
# huge lists
# TODO (johngiorgi): I need to stratify the K-folds, but sklearns implementation
# wont handle a y matrix of three dimensions, solve this!
# TODO (johngiorgi): It might be best that the outer most loop (k_folds) just
# generate test/train indicies directly
# TODO (johngiorgi): I need to name the models based on their dataset folder
# TODO (johngiorgi): https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# TODO (johngiorgi): I NEED to be able to get the per fold performance metrics. Dumb solution:
# save output of call to Saber to a file (command | tee ~/outputfile.txt or see here: https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file)
# TODO (johngiorgi): make sure this process is shuffling the data
# TODO (johngiorgi): https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/

NUM_UNITS_WORD_LSTM = 200
NUM_UNITS_CHAR_LSTM = 200
NUM_UNITS_DENSE = NUM_UNITS_WORD_LSTM // 2

class MultiTaskLSTMCRF(object):
    """ A Keras implementation of BiLSTM-CRF for sequence labeling. """

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
            model: a list of keras models, sharing (excluding crf layer) sharing
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
            word_embeddings = Embedding(input_dim=len(self.ds[0].word_type_to_idx) + 1,
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
                                                   recurrent_dropout=self.config['dropout_rate'])
        bwd_state = LSTM(NUM_UNITS_CHAR_LSTM // 2, return_state=True,
                                                   go_backwards=True,
                                                   recurrent_dropout=self.config['dropout_rate'])



        # Word-level BiLSTM
        word_BiLSTM = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                         return_sequences=True,
                                         recurrent_dropout=self.config['dropout_rate']))

        # Feedforward before CRF
        feedforward_af_word_lstm = TimeDistributed(
            Dense(units=NUM_UNITS_DENSE,
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
            char_embeddings_shared = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.config['character_embedding_dimension'])))(char_embeddings_shared)

            # Character-level BiLSTM
            fwd_state_shared = fwd_state(char_embeddings_shared)[-2]
            bwd_state_shared = bwd_state(char_embeddings_shared)[-2]
            char_embeddings_shared = Concatenate(axis=-1)([fwd_state_shared, bwd_state_shared])
            # Shape = (batch size, max sentence length, char BiLSTM hidden size)
            char_embeddings_shared = Lambda(lambda x: K.reshape(x, shape=(-1, s[1], NUM_UNITS_CHAR_LSTM)))(char_embeddings_shared)

            # Concatenate word- and char-level embeddings
            model = Concatenate(axis=-1)([word_embeddings_shared, char_embeddings_shared])

            # Dropout
            model = Dropout(self.config['dropout_rate'])(model)

            # Word-level BiLSTM
            model = word_BiLSTM(model)

            # Dropout
            model = Dropout(self.config['dropout_rate'])(model)

            # Feedforward after word-level BiLSTM
            model = feedforward_af_word_lstm(model)
            # Feedforward before CRF
            model = TimeDistributed(Dense(units=ds.tag_type_count,
                                          activation=self.config['activation_function']))(model)

            # CRF output layer
            crf = CRF(ds.tag_type_count)
            output_layer = crf(model)

            # Fully specified model
            self.model.append(Model(inputs=[word_ids, char_ids], outputs=[output_layer]))
            self.crf.append(crf)

        return self.model, self.crf

    def compile_(self):
        """Compiles a bidirectional multi-task LSTM-CRF for sequence tagging
        using Keras."""
        for model, crf in zip(self.model, self.crf):
            compile_model(model=model,
                          loss_function=crf.loss_function,
                          optimizer=self.config['optimizer'],
                          lr=self.config['learning_rate'],
                          decay=self.config['decay'],
                          clipnorm=self.config['gradient_normalization'],
                          verbose=self.config['verbose'])

    def fit_(self, checkpointer):
        """Fits a bidirectional multi-task LSTM-CRF for for sequence tagging
        using Keras. """
        # get train/valid indicies for all datasets
        train_valid_indices = self._get_train_valid_indices()

        ## FOLDS
        for fold in range(self.config['k_folds']):
            # get the train/valid partitioned data for all datasets
            data_partitions = self._get_data_partitions(train_valid_indices, fold)
            # create the Keras Callback object for computing/storing metrics
            metrics_current_fold = self._get_metrics(data_partitions)
            ## EPOCHS
            for epoch in range(self.config['maximum_number_of_epochs']):
                print('[INFO] Fold: {}; Global epoch: {}'.format(fold + 1, epoch + 1))
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
                              callbacks=[#checkpointer,
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

    def _get_train_valid_indices(self):
        """Get train and valid indicies for all k-folds for all datasets.

        For all datatsets self.ds, gets k-fold train and valid indicies
        (number of k_folds specified by self.config['k_folds']). Returns a list
        of list of two-tuples, where the outer list is of length len(self.ds),
        the inner list is of length self.config['k_folds'] and contains
        two-tuples corresponding to train indicies and valid indicies
        respectively. The train indicies for the ith dataset and jth fold would
        thus be compound_train_valid_indices[i][j][0].

        Returns:
            compound_train_valid_indices: a list of list of two-tuples, where
            compound_train_valid_indices[i][j] is a tuple containing the train
            and valid indicies (in that order) for the ith dataset and jth
            k-fold.
        """
        # acc
        compound_train_valid_indices = []
        # Sklearn KFold object
        kf = KFold(n_splits=self.config['k_folds'], random_state=42)

        for ds in self.ds:
            X = ds.train_word_idx_sequence
            # acc
            dataset_train_valid_indices = []
            for train_idx, valid_idx in kf.split(X):
                dataset_train_valid_indices.append((train_idx, valid_idx))
            compound_train_valid_indices.append(dataset_train_valid_indices)

        return compound_train_valid_indices

    def _get_data_partitions(self, train_valid_indices, fold):
        """Get train and valid partitions for all k-folds for all datasets.

        For all datasets self.ds, gets the train and valid partitions for
        all k folds (number of k_folds specified by self.config['k_folds']).
        Returns a list of six-tuples:

        (X_train_word, X_valid_word, X_train_char, X_valid_char, y_train, y_valid)

        Where X represents the inputs, and y the labels. Inputs include
        sequences of words (X_word), and sequences of characters (X_char)

        Returns:
            six-tuple containing train and valid data for all datasets.
        """
        # acc
        data_partition = []

        for i, ds in enumerate(self.ds):
            X_word = ds.train_word_idx_sequence
            X_char = ds.train_char_idx_sequence
            y = ds.train_tag_idx_sequence
            # train_valid_indices[i][fold] is a two-tuple, where index
            # 0 contains the train indicies and index 1 the valid
            # indicies
            X_word_train = X_word[train_valid_indices[i][fold][0]]
            X_word_valid = X_word[train_valid_indices[i][fold][1]]
            X_char_train = X_char[train_valid_indices[i][fold][0]]
            X_char_valid = X_char[train_valid_indices[i][fold][1]]
            y_train = y[train_valid_indices[i][fold][0]]
            y_valid = y[train_valid_indices[i][fold][1]]

            data_partition.append((X_word_train,
                                   X_word_valid,
                                   X_char_train,
                                   X_char_valid,
                                   y_train,
                                   y_valid))

        return data_partition

    def _get_metrics(self, data_partitions):
        """Creates Keras Metrics Callback objects, one for each dataset.

        Args:
            data_paritions: six-tuple containing train/valid data for all ds.
        """
        # acc
        metrics = []
        # get final part of each datasets directory, i.e. the dataset names
        ds_names = '_'.join([os.path.basename(os.path.normpath(x)) for x in self.config['dataset_folder']])

        for i, ds in enumerate(self.ds):
            # data_partitions[i] is a four-tuple where index 0 contains X_train
            # data partition, index 1 X_valid data partition, ..., for dataset i
            X_word_train = data_partitions[i][0]
            X_word_valid = data_partitions[i][1]
            X_char_train = data_partitions[i][2]
            X_char_valid = data_partitions[i][3]
            y_train = data_partitions[i][4]
            y_valid = data_partitions[i][5]

            # create a dir name with date and time
            eval = strftime("eval_%a_%b_%d_%I:%M").lower()
            # create an evaluation folder if it does not exist
            output_folder_ = os.path.join(self.config['output_folder'], ds_names, eval)
            make_dir(output_folder_)

            metrics.append(Metrics([X_word_train, X_char_train],
                                   [X_word_valid, X_char_valid],
                                   y_train, y_valid,
                                   tag_type_to_idx = ds.tag_type_to_idx,
                                   output_folder=output_folder_))

        return metrics
