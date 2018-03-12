import os

from keras import optimizers
from keras.models import Model
from keras.models import Input
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Concatenate
from keras_contrib.layers.crf import CRF

from sklearn.model_selection import KFold

import numpy as np

from metrics import Metrics
from utils_models import compile_model
from utils_generic import make_dir


# https://stackoverflow.com/questions/48615003/multi-task-learning-in-keras
# https://machinelearningmastery.com/keras-functional-api-deep-learning/
# https://medium.com/@literallywords/stratified-k-fold-with-keras-e57c487b1416

# TODO (johngiorgi): not clear if I need to clear non-shared layers when
# building the model
# TODO (johngiorgi): the way I get train/test partitions is likely copying
# huge lists
# TODO (johngiorgi): I need to stratify the K-folds, but sklearns implementation
# wont handle a y matrix of three dimensions, solve this!
# TODO (johngiorgi): consider introduction a new function, create_model()
# TODO (johngiorgi): It might be best that the outer most loop (k_folds) just
# generate test/train indicies directly
# TODO (johngiorgi): I need to name the models based on their dataset folder
# TODO (johngiorgi): https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
# TODO (johngiorgi): I NEED to be able to get the per fold performance metrics. Dumb solution:
# save output of call to Kari to a file (command | tee ~/outputfile.txt or see here: https://askubuntu.com/questions/420981/how-do-i-save-terminal-output-to-a-file)
# TODO (johngiorgi): Setup learning rate decay.
# TODO (johngiorgi): make sure this process is shuffling the data

NUM_UNITS_WORD_LSTM = 200
NUM_UNITS_DENSE = 200

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
        """ Specifies a multi-task bidirectional LSTM-CRF for sequence tagging
        using Keras.

        Implements a hybrid long short-term memory network-condition random
        field (LSTM-CRF) multi-task model for sequence tagging.

        Returns:
            model: a list of keras models, sharing (excluding crf layer) sharing
                   some number of layers.
            crf: a list of task-specific crf layers implemented using
                 keras.contrib, one for each model.
        """

        ## TOKEN EMBEDDING LAYER
        if self.token_embedding_matrix is None:
            word_embeddings = Embedding(input_dim=len(self.ds[0].word_type_to_idx),
                                        output_dim=self.config['token_embedding_dimension'],
                                        input_length=self.config['max_seq_len'],
                                        mask_zero=True)
        else:
            word_embeddings = Embedding(input_dim=len(self.ds[0].word_type_to_idx) + 1,
                                        output_dim=self.token_embedding_matrix.shape[1],
                                        weights=[self.token_embedding_matrix],
                                        input_length=self.config['max_seq_len'],
                                        mask_zero=True,
                                        trainable=(not self.config['freeze_token_embeddings']))

        ## CHAR EMBEDDING LAYER
        '''
        char_ids = Input(shape=(10, self.config['max_seq_len'], ), dtype='int32')

        char_embeddings = Embedding(
            input_dim=(len(self.ds[0].char_type_to_idx)),
            output_dim=self.character_embedding_dimension,
            mask_zero=True
        )(char_ids)

        fwd_state = LSTM(100, return_state=True)(char_embeddings)[-2]
        bwd_state = LSTM(100, return_state=True, go_backwards=True)(char_embeddings)[-2]

        char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
        '''


        ## TOKEN BILSTM LAYER
        word_BiLSTM = Bidirectional(LSTM(units=NUM_UNITS_WORD_LSTM // 2,
                                         return_sequences=True,
                                         recurrent_dropout=self.config['dropout_rate']))
        ## FULLY CONNECTED LAYER
        fully_connected = TimeDistributed(Dense(units=NUM_UNITS_DENSE // 2,
                                                activation=self.config['activation_function']))

        # specify model, taking into account the shared layers
        for ds in self.ds:
            input_layer = Input(shape=(self.config['max_seq_len'], ), dtype='int32')
            model = word_embeddings(input_layer)
            # model = Concatenate(axis=-1)([model, char_embeddings])
            model = word_BiLSTM(model)
            model = fully_connected(model)
            crf = CRF(ds.tag_type_count)
            output_layer = crf(model)
            # fully specified model
            self.model.append(Model(inputs=input_layer, outputs=output_layer))
            self.crf.append(crf)

            # clear all non-shared layers
            # input_layer = None
            # output_layer = None
            # crf = None
            # model = None

        return self.model, self.crf

    def compile_(self):
        """ Compiles a bidirectional multi-task LSTM-CRF for for sequence
        tagging using Keras. """
        for model, crf in zip(self.model, self.crf):
            compile_model(model=model,
                          learning_rate=self.config['learning_rate'],
                          optimizer=self.config['optimizer'],
                          loss_function=crf.loss_function)

    def fit_(self, checkpointer):
        """ Fits a bidirectional multi-task LSTM-CRF for for sequence tagging
        using Keras. """
        # get indices of a k fold split training set
        # for each fold in this split
        # for each epoch in global epochs
        # for each dataset and model
        # fit the model, compute metrics for train/valid sets
        # get average performance scores for this fold
        # empty the model
        # specify and compile again
        # repeate\

        # get train/valid indicies for all datasets
        train_valid_indices = self._get_train_valid_indices()

        ## FOLDS
        for fold in range(self.k_folds):
            print('[INFO] Fold: ', fold + 1)
            # get the train/valid partitioned data for all datasets
            data_partitions = self._get_data_partitions(train_valid_indices, fold)
            # create the Keras Callback object for computing/storing metrics
            self._get_metrics(data_partitions)
            ## EPOCHS
            for epoch in range(self.maximum_number_of_epochs):
                print('[INFO] Global epoch: ', epoch + 1)
                ## DATASETS/MODELS
                for i, (ds, model) in enumerate(zip(self.ds, self.model)):

                    # mainly for clearness
                    X_train = data_partitions[i][0]
                    X_valid = data_partitions[i][1]
                    y_train = data_partitions[i][2]
                    y_valid = data_partitions[i][3]

                    model.fit(x=X_train,
                              y=y_train,
                              batch_size=self.batch_size,
                              epochs=1,
                              callbacks=[
                                checkpointer,
                                self.metrics[i]],
                              validation_data=[X_valid, y_valid],
                              verbose=1)

            # end of a k-fold, so clear the model, specify and compile again
            self.model = []
            self.crf = []
            self.metrics = []
            self.specify_()
            self.compile_()

    def _get_train_valid_indices(self):
        """
        Get train and valid indicies for all k-folds for all datasets.

        For all datatsets self.ds, gets k-fold train and valid indicies
        (number of k_folds specified by self.k_folds). Returns a list
        of list of two-tuples, where the outer list is of length len(self.ds),
        the inner list is of length len(self.k_folds) and contains two-tuples
        corresponding to train indicies and valid indicies respectively. The
        train indicies for the ith dataset and jth fold would thus be
        compound_train_valid_indices[i][j][0].

        Returns:
            compound_train_valid_indices: a list of list of two-tuples, where
            compound_train_valid_indices[i][j] is a tuple containing the train
            and valid indicies (in that order) for the ith dataset and jth
            k-fold.
        """
        # acc
        compound_train_valid_indices = []
        # Sklearn KFold object
        kf = KFold(n_splits=self.k_folds, random_state=42)

        for ds in self.ds:
            X = ds.train_word_idx_sequence
            # acc
            dataset_train_valid_indices = []
            for train_idx, valid_idx in kf.split(X):
                dataset_train_valid_indices.append((train_idx, valid_idx))
            compound_train_valid_indices.append(dataset_train_valid_indices)

        return compound_train_valid_indices

    def _get_data_partitions(self, train_valid_indices, fold):
        """ Get train and valid partitions for all k-folds for all datasets.

        For all datasets self.ds, gets the train and valid partitions for
        all k folds (number of k_folds specified by self.k_folds). Returns a
        list of four-tuples, (X_train, X_valid, y_train, y_valid)
        """
        # acc
        data_partition = []

        for i, ds in enumerate(self.ds):
            X = ds.train_word_idx_sequence
            y = ds.train_tag_idx_sequence
            # train_valid_indices[i][fold] is a two-tuple, where index
            # 0 contains the train indicies and index 1 the valid
            # indicies
            X_train = X[train_valid_indices[i][fold][0]]
            X_valid = X[train_valid_indices[i][fold][1]]
            y_train = y[train_valid_indices[i][fold][0]]
            y_valid = y[train_valid_indices[i][fold][1]]

            data_partition.append((X_train, X_valid, y_train, y_valid))

        return data_partition

    def _get_metrics(self, data_partitions):
        """
        """
        # acc
        metrics = []

        for i, ds in enumerate(self.ds):
            # data_partitions[i] is a four-tuple where index 0 contains  X_train
            # data partition, index 1 X_valid data partition, ..., for dataset i
            X_train = data_partitions[i][0]
            X_valid = data_partitions[i][1]
            y_train = data_partitions[i][2]
            y_valid = data_partitions[i][3]

            # get final part of dataset folder path, i.e. the dataset name
            ds_name = os.path.basename(os.path.normpath(self.dataset_folder[0]))
            # create an evaluation folder if it does not exist
            output_folder_ = os.path.join(self.output_folder, ds_name, 'eval')
            make_dir(output_folder_)

            metrics.append(Metrics(X_train, X_valid, y_train, y_valid,
                                   tag_type_to_idx = ds.tag_type_to_idx))

        self.metrics = metrics
